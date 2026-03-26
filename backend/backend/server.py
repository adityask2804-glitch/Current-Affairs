from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks, Body
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from bson import ObjectId
import asyncio

# Scraping and AI imports
import feedparser
from newspaper import Article
from bs4 import BeautifulSoup
import requests
from emergentintegrations.llm.chat import LlmChat, UserMessage

# PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from io import BytesIO
from fastapi.responses import StreamingResponse

# Background scheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# LLM API Key
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')

# Create the main app
app = FastAPI()
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Topic categories for UPSC
TOPIC_CATEGORIES = [
    "Polity & Governance",
    "Economy",
    "International Relations",
    "Environment & Ecology",
    "Science & Technology",
    "Social Issues",
    "Miscellaneous"
]

# RSS Feeds (using accessible sources)
RSS_FEEDS = {
    "The Hindu - Opinion": "https://www.thehindu.com/opinion/feeder/default.rss",
    "The Hindu - Editorial": "https://www.thehindu.com/opinion/editorial/feeder/default.rss",
    "Indian Express - Opinion": "https://indianexpress.com/section/opinion/feed/",
    "Indian Express - Explained": "https://indianexpress.com/section/explained/feed/"
}

# ==================== Models ====================

class ArticleModel(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    url: str
    title: str
    content: str
    source: str
    published_date: Optional[datetime] = None
    fetched_date: datetime = Field(default_factory=datetime.utcnow)
    category: Optional[str] = None
    
    class Config:
        populate_by_name = True

class SummaryModel(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    article_id: str
    article_title: str
    article_url: str
    summary_text: str
    topic_category: str
    source: str
    created_date: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True

class DailySummaryResponse(BaseModel):
    date: str
    summaries_by_topic: Dict[str, List[SummaryModel]]
    total_count: int

class PushTokenModel(BaseModel):
    token: str
    device_type: str
    registered_at: datetime = Field(default_factory=datetime.utcnow)

# ==================== Helper Functions ====================

async def categorize_article_with_llm(title: str, content: str) -> str:
    """Use LLM to categorize article into UPSC topics"""
    try:
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"categorize_{datetime.utcnow().timestamp()}",
            system_message="You are an expert in UPSC Civil Services Examination preparation and current affairs categorization."
        ).with_model("openai", "gpt-5.2")
        
        categories_str = ", ".join(TOPIC_CATEGORIES)
        prompt = f"""Analyze this article and categorize it into ONE of these UPSC-relevant topics:
{categories_str}

Article Title: {title}
Article Content (first 500 chars): {content[:500]}

Respond with ONLY the category name, nothing else."""
        
        message = UserMessage(text=prompt)
        response = await chat.send_message(message)
        
        category = response.strip()
        if category not in TOPIC_CATEGORIES:
            category = "Miscellaneous"
        
        return category
    except Exception as e:
        logger.error(f"Error in categorization: {e}")
        return "Miscellaneous"

async def summarize_article_with_llm(title: str, content: str) -> str:
    """Use LLM to create UPSC-focused summary"""
    try:
        chat = LlmChat(
            api_key=EMERGENT_LLM_KEY,
            session_id=f"summarize_{datetime.utcnow().timestamp()}",
            system_message="You are an expert in creating concise, UPSC-focused summaries of current affairs articles."
        ).with_model("openai", "gpt-5.2")
        
        prompt = f"""Create a concise summary of this article for UPSC Civil Services preparation. 
Focus on:
1. Key facts and figures
2. Government policies/schemes mentioned
3. Constitutional/legal aspects
4. International relations implications
5. Environmental/social impacts
6. Economic significance

Keep the summary between 150-250 words, clear and exam-oriented.

Article Title: {title}
Article Content: {content}

Summary:"""
        
        message = UserMessage(text=prompt)
        response = await chat.send_message(message)
        
        return response.strip()
    except Exception as e:
        logger.error(f"Error in summarization: {e}")
        return f"Summary generation failed. Original article: {content[:300]}..."

async def fetch_article_content(url: str) -> tuple:
    """Fetch full article content using newspaper3k"""
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        return article.title, article.text, article.publish_date
    except Exception as e:
        logger.error(f"Error fetching article from {url}: {e}")
        return None, None, None

async def scrape_and_process_articles():
    """Main scraping function - fetches articles from RSS feeds and processes them"""
    logger.info("Starting article scraping and processing...")
    
    processed_count = 0
    
    for source_name, feed_url in RSS_FEEDS.items():
        try:
            logger.info(f"Fetching from {source_name}...")
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:15]:  # Process top 15 articles from each feed for comprehensive coverage
                try:
                    url = entry.link
                    
                    # Check if article already exists
                    existing = await db.articles.find_one({"url": url})
                    if existing:
                        logger.info(f"Article already exists: {url}")
                        continue
                    
                    # Fetch full article content
                    title, content, pub_date = await fetch_article_content(url)
                    
                    if not title or not content or len(content) < 100:
                        logger.warning(f"Skipping article with insufficient content: {url}")
                        continue
                    
                    # Categorize article
                    category = await categorize_article_with_llm(title, content)
                    
                    # Save article
                    article_data = {
                        "url": url,
                        "title": title,
                        "content": content,
                        "source": source_name,
                        "published_date": pub_date,
                        "fetched_date": datetime.utcnow(),
                        "category": category
                    }
                    
                    result = await db.articles.insert_one(article_data)
                    article_id = str(result.inserted_id)
                    
                    # Generate summary
                    summary_text = await summarize_article_with_llm(title, content)
                    
                    # Save summary
                    summary_data = {
                        "article_id": article_id,
                        "article_title": title,
                        "article_url": url,
                        "summary_text": summary_text,
                        "topic_category": category,
                        "source": source_name,
                        "created_date": datetime.utcnow()
                    }
                    
                    await db.summaries.insert_one(summary_data)
                    processed_count += 1
                    
                    logger.info(f"Processed: {title} [{category}]")
                    
                    # Small delay to avoid overwhelming the servers and ensure quality processing
                    await asyncio.sleep(3)
                    
                except Exception as e:
                    logger.error(f"Error processing entry: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error fetching feed {source_name}: {e}")
            continue
    
    logger.info(f"Scraping completed. Processed {processed_count} new articles.")
    return processed_count

def generate_pdf_for_topic(summaries: List[dict], topic: str) -> BytesIO:
    """Generate PDF for a specific topic"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor='darkblue',
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor='darkgreen',
        spaceAfter=12,
        spaceBefore=12
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['BodyText'],
        fontSize=11,
        alignment=TA_JUSTIFY,
        spaceAfter=12
    )
    
    story = []
    
    # Title
    story.append(Paragraph(f"UPSC Current Affairs - {topic}", title_style))
    story.append(Paragraph(f"Generated on: {datetime.utcnow().strftime('%d %B %Y')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Add summaries
    for idx, summary in enumerate(summaries, 1):
        # Article title
        story.append(Paragraph(f"{idx}. {summary['article_title']}", heading_style))
        
        # Source and date
        source_text = f"<i>Source: {summary['source']} | Date: {summary['created_date'].strftime('%d %B %Y')}</i>"
        story.append(Paragraph(source_text, styles['Italic']))
        story.append(Spacer(1, 0.1*inch))
        
        # Summary content
        story.append(Paragraph(summary['summary_text'], body_style))
        
        # URL
        url_text = f"<i>Read full article: {summary['article_url']}</i>"
        story.append(Paragraph(url_text, styles['Italic']))
        story.append(Spacer(1, 0.2*inch))
        
        # Page break after every 3 articles
        if idx % 3 == 0 and idx < len(summaries):
            story.append(PageBreak())
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# ==================== API Endpoints ====================

@api_router.get("/")
async def root():
    return {"message": "UPSC Current Affairs API", "version": "1.0"}

@api_router.post("/scrape")
async def trigger_scrape(background_tasks: BackgroundTasks):
    """Manually trigger article scraping"""
    background_tasks.add_task(scrape_and_process_articles)
    return {"message": "Scraping started in background", "status": "processing"}

@api_router.get("/daily-summary")
async def get_daily_summary(date: Optional[str] = None):
    """Get daily summaries organized by topic"""
    try:
        # Parse date or use today
        if date:
            target_date = datetime.strptime(date, "%Y-%m-%d")
        else:
            target_date = datetime.utcnow()
        
        # Get summaries from the target date
        start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        
        summaries_cursor = db.summaries.find({
            "created_date": {"$gte": start_of_day, "$lt": end_of_day}
        })
        
        summaries = await summaries_cursor.to_list(length=1000)
        
        # Organize by topic
        summaries_by_topic = {topic: [] for topic in TOPIC_CATEGORIES}
        
        for summary in summaries:
            summary['_id'] = str(summary['_id'])
            topic = summary.get('topic_category', 'Miscellaneous')
            if topic in summaries_by_topic:
                summaries_by_topic[topic].append(summary)
            else:
                summaries_by_topic['Miscellaneous'].append(summary)
        
        return {
            "date": target_date.strftime("%Y-%m-%d"),
            "summaries_by_topic": summaries_by_topic,
            "total_count": len(summaries)
        }
    
    except Exception as e:
        logger.error(f"Error fetching daily summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/summaries/{topic}")
async def get_summaries_by_topic(topic: str, limit: int = 50):
    """Get all summaries for a specific topic"""
    try:
        if topic not in TOPIC_CATEGORIES:
            raise HTTPException(status_code=400, detail="Invalid topic category")
        
        summaries_cursor = db.summaries.find({
            "topic_category": topic
        }).sort("created_date", -1).limit(limit)
        
        summaries = await summaries_cursor.to_list(length=limit)
        
        for summary in summaries:
            summary['_id'] = str(summary['_id'])
        
        return {
            "topic": topic,
            "count": len(summaries),
            "summaries": summaries
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching summaries for topic {topic}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/topics")
async def get_all_topics():
    """Get all topics with summary counts"""
    try:
        topic_counts = []
        
        for topic in TOPIC_CATEGORIES:
            count = await db.summaries.count_documents({"topic_category": topic})
            topic_counts.append({
                "topic": topic,
                "count": count
            })
        
        return {"topics": topic_counts}
    
    except Exception as e:
        logger.error(f"Error fetching topics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/download-pdf/{topic}")
async def download_topic_pdf(topic: str, ids: Optional[str] = None):
    """Download PDF of summaries for a specific topic"""
    try:
        if topic not in TOPIC_CATEGORIES:
            raise HTTPException(status_code=400, detail="Invalid topic category")
        
        # Parse IDs from query string
        summary_ids = []
        if ids:
            summary_ids = [id.strip() for id in ids.split(',') if id.strip()]
        
        # Fetch summaries - either selected IDs or all from topic
        if summary_ids and len(summary_ids) > 0:
            # Fetch only selected summaries
            from bson import ObjectId
            object_ids = [ObjectId(id) for id in summary_ids if ObjectId.is_valid(id)]
            summaries_cursor = db.summaries.find({
                "_id": {"$in": object_ids},
                "topic_category": topic
            }).sort("created_date", -1)
            summaries = await summaries_cursor.to_list(length=100)
        else:
            # Fetch all summaries for the topic
            summaries_cursor = db.summaries.find({
                "topic_category": topic
            }).sort("created_date", -1).limit(100)
            summaries = await summaries_cursor.to_list(length=100)
        
        if not summaries:
            raise HTTPException(status_code=404, detail="No summaries found")
        
        # Generate PDF
        pdf_buffer = generate_pdf_for_topic(summaries, topic)
        
        # Return as downloadable file
        filename = f"UPSC_{topic.replace(' ', '_')}_{datetime.utcnow().strftime('%Y%m%d')}.pdf"
        
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "application/pdf",
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/register-push-token")
async def register_push_token(token_data: PushTokenModel):
    """Register device token for push notifications"""
    try:
        # Check if token already exists
        existing = await db.push_tokens.find_one({"token": token_data.token})
        
        if existing:
            return {"message": "Token already registered", "status": "exists"}
        
        # Save new token
        await db.push_tokens.insert_one(token_data.dict())
        
        return {"message": "Token registered successfully", "status": "registered"}
    
    except Exception as e:
        logger.error(f"Error registering push token: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/stats")
async def get_stats():
    """Get overall statistics"""
    try:
        total_articles = await db.articles.count_documents({})
        total_summaries = await db.summaries.count_documents({})
        
        # Get recent summaries
        recent = await db.summaries.find().sort("created_date", -1).limit(5).to_list(5)
        
        for item in recent:
            item['_id'] = str(item['_id'])
        
        return {
            "total_articles": total_articles,
            "total_summaries": total_summaries,
            "recent_summaries": recent
        }
    
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Include router
app.include_router(api_router)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Background Scheduler ====================

scheduler = BackgroundScheduler()

def scheduled_scrape_job():
    """Job to run daily scraping"""
    logger.info("Running scheduled scraping job...")
    asyncio.run(scrape_and_process_articles())

# Schedule daily scraping at 7:30 AM for comprehensive article processing
scheduler.add_job(
    scheduled_scrape_job,
    CronTrigger(hour=7, minute=30),
    id='daily_scrape',
    name='Daily article scraping',
    replace_existing=True
)

@app.on_event("startup")
async def startup_event():
    """Start scheduler on app startup"""
    scheduler.start()
    logger.info("Background scheduler started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    scheduler.shutdown()
    client.close()
    logger.info("Application shutdown complete")
