import os
import re
import json
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from dotenv import load_dotenv


load_dotenv()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reddit_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class RedditPost:
    """Data class for Reddit posts and comments."""
    title: str
    content: str
    subreddit: str
    url: str
    timestamp: str
    upvotes: int
    post_type: str  

@dataclass
class PersonalityTraits:
    """Data class for MBTI-style personality traits (as percentages)."""
    introversion: int  
    intuition: int     
    feeling: int       
    perceiving: int    


@dataclass
class Motivations:
    """Data class for user motivations with intensity scores."""
    primary_motivation: str
    secondary_motivation: str
    intensity_scores: Dict[str, int]  


class UserPersonaModel(BaseModel):
    """Enhanced Pydantic model for user persona with validation."""
    name: str = Field(description="Inferred or generic name for the user")
    age_range: str = Field(description="Estimated age range (e.g., 25-35)")
    location: str = Field(description="Inferred location or 'Unknown'")
    occupation: str = Field(description="Inferred occupation or industry")
    
   
    status: str = Field(description="Social/professional status (e.g., 'Student', 'Professional', 'Freelancer')")
    tier: str = Field(description="Experience/skill tier (e.g., 'Beginner', 'Intermediate', 'Advanced', 'Expert')")
    archetype: str = Field(description="User archetype (e.g., 'The Explorer', 'The Creator', 'The Helper')")
    
  
    interests: List[str] = Field(description="List of user interests")
    personality_traits: List[str] = Field(description="List of personality traits")
    goals: List[str] = Field(description="List of apparent goals")
    frustrations: List[str] = Field(description="List of frustrations")
    preferred_subreddits: List[str] = Field(description="List of frequented subreddits")
    communication_style: str = Field(description="Description of communication style")
    technology_comfort: str = Field(description="Level of technology comfort")
    social_media_behavior: str = Field(description="Description of social media behavior")
    
  
    motivations: Dict[str, int] = Field(description="Primary motivations with intensity scores (0-100)")
    behavior_habits: List[str] = Field(description="Observable behavior patterns and habits")
    personality_percentages: Dict[str, int] = Field(description="MBTI-style personality percentages")
    
    citations: Dict[str, List[str]] = Field(description="Citations for each characteristic")


@dataclass
class UserPersona:
    """Enhanced data class for user persona information."""
    name: str
    age_range: str
    location: str
    occupation: str
    status: str
    tier: str
    archetype: str
    interests: List[str]
    personality_traits: List[str]
    goals: List[str]
    frustrations: List[str]
    preferred_subreddits: List[str]
    communication_style: str
    technology_comfort: str
    social_media_behavior: str
    motivations: Dict[str, int]
    behavior_habits: List[str]
    personality_percentages: Dict[str, int]
    citations: Dict[str, List[str]]


class PersonaOutputParser(BaseOutputParser[UserPersonaModel]):
    """Custom parser for persona generation output using Pydantic v2."""
    
    def parse(self, text: str) -> UserPersonaModel:
        """Parse the LLM output into structured persona data."""
        try:
         
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return UserPersonaModel.model_validate(data)
            else:
               
                return self._fallback_parse(text)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"JSON parsing failed: {e}. Using fallback parsing.")
            return self._fallback_parse(text)
    
    def _fallback_parse(self, text: str) -> UserPersonaModel:
        """Fallback parsing method."""
        logger.warning("Using fallback parsing for persona data")
        return UserPersonaModel(
            name="Unknown User",
            age_range="Unknown",
            location="Unknown",
            occupation="Unknown",
            status="Unknown",
            tier="Unknown",
            archetype="Unknown",
            interests=["Unknown"],
            personality_traits=["Unknown"],
            goals=["Unknown"],
            frustrations=["Unknown"],
            preferred_subreddits=["Unknown"],
            communication_style="Unknown",
            technology_comfort="Unknown",
            social_media_behavior="Unknown",
            motivations={"unknown": 50},
            behavior_habits=["Unknown"],
            personality_percentages={"introversion": 50, "intuition": 50, "feeling": 50, "perceiving": 50},
            citations={}
        )


class RedditScraper:
    """Reddit profile scraper class."""
    
    def __init__(self, delay: float = 2.0):
        """Initialize the scraper with rate limiting."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        self.delay = delay
        
    def scrape_profile(self, profile_url: str, max_posts: int = 50) -> List[RedditPost]:
        """
        Scrape Reddit profile for posts and comments.
        
        Args:
            profile_url: Reddit profile URL
            max_posts: Maximum number of posts/comments to scrape
            
        Returns:
            List of RedditPost objects
        """
        logger.info(f"Starting to scrape profile: {profile_url}")
        
        
        if not self._is_valid_reddit_url(profile_url):
            raise ValueError(f"Invalid Reddit profile URL: {profile_url}")
        
        posts = []
        
        try:
         
            posts_url = f"{profile_url.rstrip('/')}/submitted/"
            posts.extend(self._scrape_content(posts_url, "post", max_posts // 2))
            
           
            comments_url = f"{profile_url.rstrip('/')}/comments/"
            posts.extend(self._scrape_content(comments_url, "comment", max_posts // 2))
            
        except Exception as e:
            logger.error(f"Error scraping profile: {e}")
            raise
        
        logger.info(f"Successfully scraped {len(posts)} posts/comments")
        return posts
    
    def _is_valid_reddit_url(self, url: str) -> bool:
        """Validate Reddit profile URL."""
        parsed = urlparse(url)
        return (
            parsed.netloc in ['www.reddit.com', 'reddit.com', 'old.reddit.com'] and
            '/user/' in parsed.path
        )
    
    def _scrape_content(self, url: str, content_type: str, max_items: int) -> List[RedditPost]:
        """Scrape posts or comments from a specific URL."""
        posts = []
        
        try:
           
            json_url = f"{url.rstrip('/')}.json"
            
            response = self.session.get(json_url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
           
            if isinstance(data, dict) and 'data' in data:
                items = data['data']['children']
            elif isinstance(data, list):
                items = []
                for item in data:
                    if isinstance(item, dict) and 'data' in item:
                        items.extend(item['data']['children'])
            else:
                items = []
            
            for item in items[:max_items]:
                try:
                    post_data = item['data']
                    
                    
                    post = RedditPost(
                        title=post_data.get('title', post_data.get('link_title', 'No Title')),
                        content=self._clean_content(post_data.get('selftext', post_data.get('body', ''))),
                        subreddit=post_data.get('subreddit', 'Unknown'),
                        url=f"https://reddit.com{post_data.get('permalink', '')}",
                        timestamp=datetime.fromtimestamp(
                            post_data.get('created_utc', 0)
                        ).strftime('%Y-%m-%d %H:%M:%S'),
                        upvotes=post_data.get('score', 0),
                        post_type=content_type
                    )
                    
                    
                    if post.content.strip() and len(post.content) > 10:
                        posts.append(post)
                        
                except Exception as e:
                    logger.warning(f"Error parsing {content_type}: {e}")
                    continue
            
            time.sleep(self.delay)  
            
        except Exception as e:
            logger.error(f"Error scraping {content_type} from {url}: {e}")
        
        return posts
    
    def _clean_content(self, content: str) -> str:
        """Clean and format content text."""
        if not content:
            return ""
        
       
        content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  
        content = re.sub(r'\*(.*?)\*', r'\1', content)      
        content = re.sub(r'~~(.*?)~~', r'\1', content)     
        content = re.sub(r'(.*?)', r'\1', content)        
        
        
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = content.strip()
        
        return content


class PersonaGenerator:
    """Generate user personas using LangChain v0.3 and Groq."""
    
    def __init__(self, groq_api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        """Initialize the persona generator with latest LangChain patterns."""
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model=model_name,
            temperature=0.1,
            max_tokens=4096,
            timeout=60,
            max_retries=3
        )
        
        self.parser = PersonaOutputParser()
        self._setup_chain()
    
    def _setup_chain(self):
        """Setup the LangChain chain for persona generation using v0.3 patterns."""
        
       
        system_prompt = """
        You are an expert user researcher and persona analyst. Your task is to analyze Reddit posts and comments to create a comprehensive, enhanced user persona.

        Based on the provided Reddit data, you must:
        1. Analyze the content for patterns, interests, communication style, and behavior
        2. Make reasonable inferences about demographics, psychographics, and personality traits
        3. Provide specific citations from the posts/comments for each characteristic
        4. Be respectful and avoid negative judgments
        5. Focus on constructive insights for understanding the user
        6. Assign personality percentages based on MBTI-style dimensions
        7. Rate motivations with intensity scores (0-100)

        Return your analysis in VALID JSON format with the exact structure specified in the human message.
        """
        
       
        human_prompt = """
        Analyze the following Reddit posts and comments to create a detailed, enhanced user persona:

        Reddit Data:
        {reddit_data}

        Create a comprehensive user persona and return it in this EXACT JSON format:
        {{
            "name": "Inferred or generic name (e.g., 'Tech Professional', 'Gaming Enthusiast')",
            "age_range": "Estimated age range (e.g., '25-35', '18-25')",
            "location": "Inferred location or 'Unknown'",
            "occupation": "Inferred occupation or industry",
            "status": "Social/professional status (e.g., 'Student', 'Professional', 'Freelancer', 'Entrepreneur')",
            "tier": "Experience/skill tier (e.g., 'Beginner', 'Intermediate', 'Advanced', 'Expert')",
            "archetype": "User archetype (e.g., 'The Explorer', 'The Creator', 'The Helper', 'The Achiever', 'The Sage')",
            "interests": ["list", "of", "interests", "based", "on", "posts"],
            "personality_traits": ["list", "of", "personality", "traits"],
            "goals": ["list", "of", "apparent", "goals"],
            "frustrations": ["list", "of", "frustrations"],
            "preferred_subreddits": ["list", "of", "frequented", "subreddits"],
            "communication_style": "Description of how they communicate",
            "technology_comfort": "Level of technology comfort (Low/Medium/High/Expert)",
            "social_media_behavior": "Description of their social media behavior",
            "motivations": {{
                "achievement": 75,
                "social_connection": 60,
                "knowledge_seeking": 85,
                "creative_expression": 40,
                "helping_others": 70,
                "recognition": 30
            }},
            "behavior_habits": ["list", "of", "observable", "behavior", "patterns"],
            "personality_percentages": {{
                "introversion": 65,
                "intuition": 80,
                "feeling": 45,
                "perceiving": 70
            }},
            "citations": {{
                "interests": ["Specific quote: 'quote from post/comment'"],
                "personality_traits": ["Specific quote: 'quote from post/comment'"],
                "goals": ["Specific quote: 'quote from post/comment'"],
                "frustrations": ["Specific quote: 'quote from post/comment'"],
                "occupation": ["Specific quote: 'quote from post/comment'"],
                "location": ["Specific quote: 'quote from post/comment'"],
                "age_range": ["Specific quote: 'quote from post/comment'"],
                "status": ["Specific quote: 'quote from post/comment'"],
                "tier": ["Specific quote: 'quote from post/comment'"],
                "archetype": ["Specific quote: 'quote from post/comment'"],
                "communication_style": ["Specific quote: 'quote from post/comment'"],
                "technology_comfort": ["Specific quote: 'quote from post/comment'"],
                "social_media_behavior": ["Specific quote: 'quote from post/comment'"],
                "motivations": ["Specific quote: 'quote from post/comment'"],
                "behavior_habits": ["Specific quote: 'quote from post/comment'"],
                "personality_percentages": ["Specific quote: 'quote from post/comment'"]
            }}
        }}

        Important guidelines for enhanced sections:
        
        STATUS: Infer from posts about work, education, life stage (Student, Professional, Freelancer, Entrepreneur, Retiree, etc.)
        
        TIER: Assess expertise level in their main domains (Beginner, Intermediate, Advanced, Expert)
        
        ARCHETYPE: Choose from common user archetypes:
        - The Explorer (curious, adventurous)
        - The Creator (creative, innovative)
        - The Helper (supportive, nurturing)
        - The Achiever (goal-oriented, competitive)
        - The Sage (wise, knowledge-seeking)
        - The Rebel (unconventional, challenging)
        - The Caregiver (empathetic, protective)
        
        MOTIVATIONS: Rate intensity (0-100) for:
        - achievement (accomplishing goals)
        - social_connection (building relationships)
        - knowledge_seeking (learning, understanding)
        - creative_expression (creating, innovating)
        - helping_others (supporting, teaching)
        - recognition (fame, acknowledgment)
        
        PERSONALITY PERCENTAGES (0-100):
        - introversion: 0=extremely extroverted, 100=extremely introverted
        - intuition: 0=extremely sensing/practical, 100=extremely intuitive/abstract
        - feeling: 0=extremely thinking/logical, 100=extremely feeling/emotional
        - perceiving: 0=extremely judging/structured, 100=extremely perceiving/flexible
        
        BEHAVIOR & HABITS: Observable patterns like posting frequency, interaction style, topic preferences, etc.

        Base ALL inferences on actual content from the posts/comments
        Provide specific quotes as citations for each characteristic
        Use "Unknown" if information cannot be inferred
        Keep quotes under 100 characters when possible
        Ensure JSON is valid and properly formatted
        """
        
     
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", human_prompt)
        ])
        
       
        self.chain = (
            {"reddit_data": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | self.parser
        )
    
    def generate_persona(self, posts: List[RedditPost]) -> UserPersona:
        """Generate a user persona from Reddit posts."""
        logger.info("Generating enhanced user persona...")
        
        
        reddit_data = self._format_posts_for_llm(posts)
        
        try:
            
            result = self.chain.invoke(reddit_data)
            
           
            persona = UserPersona(
                name=result.name,
                age_range=result.age_range,
                location=result.location,
                occupation=result.occupation,
                status=result.status,
                tier=result.tier,
                archetype=result.archetype,
                interests=result.interests,
                personality_traits=result.personality_traits,
                goals=result.goals,
                frustrations=result.frustrations,
                preferred_subreddits=result.preferred_subreddits,
                communication_style=result.communication_style,
                technology_comfort=result.technology_comfort,
                social_media_behavior=result.social_media_behavior,
                motivations=result.motivations,
                behavior_habits=result.behavior_habits,
                personality_percentages=result.personality_percentages,
                citations=result.citations
            )
            
            logger.info("Enhanced user persona generated successfully")
            return persona
            
        except Exception as e:
            logger.error(f"Error generating persona: {e}")
            raise
    
    def _format_posts_for_llm(self, posts: List[RedditPost]) -> str:
        """Format posts for LLM input."""
        formatted_posts = []
        
        for i, post in enumerate(posts, 1):
           
            content = post.content[:800] + "..." if len(post.content) > 800 else post.content
            
            formatted_post = f"""
            === {post.post_type.upper()} {i} ===
            Title: {post.title}
            Subreddit: r/{post.subreddit}
            Content: {content}
            Timestamp: {post.timestamp}
            Upvotes: {post.upvotes}
            URL: {post.url}
            """
            formatted_posts.append(formatted_post)
        
        return "\n".join(formatted_posts)


class PersonaWriter:
    """Write enhanced persona to text file with improved formatting."""
    
    def write_persona_to_file(self, persona: UserPersona, username: str, output_dir: str = "output"):
        """Write enhanced persona to a text file with comprehensive formatting."""
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{username}_persona.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"USER PERSONA FOR REDDIT USER: {username}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            
          
            f.write("BASIC DEMOGRAPHICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Name: {persona.name}\n")
            f.write(f"Age Range: {persona.age_range}\n")
            f.write(f"Location: {persona.location}\n")
            f.write(f"Occupation: {persona.occupation}\n")
            f.write(f"Status: {persona.status}\n")
            f.write(f"Tier: {persona.tier}\n")
            f.write(f"Archetype: {persona.archetype}\n\n")
            
          
            f.write("INTERESTS\n")
            f.write("-" * 20 + "\n")
            for interest in persona.interests:
                f.write(f"• {interest}\n")
            f.write("\n")
            
            
            f.write("PERSONALITY TRAITS\n")
            f.write("-" * 20 + "\n")
            for trait in persona.personality_traits:
                f.write(f"• {trait}\n")
            f.write("\n")
            
           
            f.write("PERSONALITY ASSESSMENT (MBTI-Style)\n")
            f.write("-" * 35 + "\n")
            f.write(f"Introversion: {persona.personality_percentages.get('introversion', 50)}%\n")
            f.write(f"  (0% = Extremely Extroverted, 100% = Extremely Introverted)\n\n")
            f.write(f"Intuition: {persona.personality_percentages.get('intuition', 50)}%\n")
            f.write(f"  (0% = Extremely Sensing, 100% = Extremely Intuitive)\n\n")
            f.write(f"Feeling: {persona.personality_percentages.get('feeling', 50)}%\n")
            f.write(f"  (0% = Extremely Thinking, 100% = Extremely Feeling)\n\n")
            f.write(f"Perceiving: {persona.personality_percentages.get('perceiving', 50)}%\n")
            f.write(f"  (0% = Extremely Judging, 100% = Extremely Perceiving)\n\n")
            
          
            f.write("GOALS & ASPIRATIONS\n")
            f.write("-" * 20 + "\n")
            for goal in persona.goals:
                f.write(f"• {goal}\n")
            f.write("\n")
            
           
            f.write("MOTIVATIONS (Intensity Scores)\n")
            f.write("-" * 30 + "\n")
            for motivation, intensity in persona.motivations.items():
                f.write(f"• {motivation.replace('_', ' ').title()}: {intensity}/100\n")
            f.write("\n")
            
            
            f.write("FRUSTRATIONS & PAIN POINTS\n")
            f.write("-" * 30 + "\n")
            for frustration in persona.frustrations:
                f.write(f"• {frustration}\n")
            f.write("\n")
            
          
            f.write("BEHAVIOR & HABITS\n")
            f.write("-" * 20 + "\n")
            for habit in persona.behavior_habits:
                f.write(f"• {habit}\n")
            f.write("\n")
            
         
            f.write("PREFERRED SUBREDDITS\n")
            f.write("-" * 20 + "\n")
            for subreddit in persona.preferred_subreddits:
                f.write(f"• r/{subreddit}\n")
            f.write("\n")
            
           
            f.write("BEHAVIORAL INSIGHTS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Communication Style: {persona.communication_style}\n\n")
            f.write(f"Technology Comfort: {persona.technology_comfort}\n\n")
            f.write(f"Social Media Behavior: {persona.social_media_behavior}\n\n")
            
          
            f.write("SUPPORTING EVIDENCE & CITATIONS\n")
            f.write("=" * 35 + "\n")
            f.write("The following quotes from the user's posts and comments support each characteristic:\n\n")
            
            for category, citations in persona.citations.items():
                if citations:  
                    f.write(f"{category.upper().replace('_', ' ')}:\n")
                    for citation in citations:
                        f.write(f"  • {citation}\n")
                    f.write("\n")
        
        logger.info(f"Enhanced persona written to {filepath}")
        return filepath


def process_single_user(profile_url: str, groq_api_key: str) -> str:
    """Process a single Reddit user and generate their enhanced persona."""
  
    username = profile_url.split('/user/')[-1].rstrip('/')
    
    print(f"\nProcessing user: {username}")
    print("-" * 50)
    
    
    scraper = RedditScraper(delay=2.0)
    generator = PersonaGenerator(groq_api_key)
    writer = PersonaWriter()
    
    try:
       
        posts = scraper.scrape_profile(profile_url, max_posts=50)
        
        if not posts:
            print(f"No posts found for user {username}")
            return ""
        
        print(f"Found {len(posts)} posts/comments")
        
       
        persona = generator.generate_persona(posts)
        
       
        filepath = writer.write_persona_to_file(persona, username)
        
        print(f"✓ Enhanced persona generated and saved to: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error processing {profile_url}: {e}")
        print(f"✗ Error processing {username}: {e}")
        return ""


def main():
    """Main function to run the enhanced Reddit persona generator."""
    parser = argparse.ArgumentParser(description='Generate enhanced Reddit user personas')
    parser.add_argument('--url', '-u', type=str, help='Reddit profile URL to analyze')
    parser.add_argument('--max-posts', '-m', type=int, default=50, help='Maximum posts to scrape (default: 50)')
    
    args = parser.parse_args()
    
    
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        print("Error: GROQ_API_KEY environment variable is required")
        print("Please set it in your .env file or environment")
        return
    
   
    if args.url:
        
        profile_urls = [args.url]
    else:
        
        profile_url = input("Enter Reddit profile URL (or press Enter for default examples): ").strip()
        
        if profile_url:
            profile_urls = [profile_url]
        else:
            
            profile_urls = [
                "https://www.reddit.com/user/kojied/",
                "https://www.reddit.com/user/Hungry-Move-6603/"
            ]
    
    print("Starting Enhanced Reddit User Persona Generator")
    print(f"Processing {len(profile_urls)} profiles...")
    print("=" * 60)
    
    successful_processes = 0
    
    for profile_url in profile_urls:
        filepath = process_single_user(profile_url, groq_api_key)
        if filepath:
            successful_processes += 1
    
    print("\n" + "=" * 60)
    print(f"Successfully processed {successful_processes}/{len(profile_urls)} profiles")
    print("Check the 'output' directory for generated persona files")
    print("Check 'reddit_scraper.log' for detailed logs")
    
    print("\nEnhanced Features Added:")
    print("• Status (professional/social status)")
    print("• Tier (experience level)")
    print("• Archetype (user personality archetype)")
    print("• Motivations (with intensity scores 0-100)")
    print("• Behavior & Habits (observable patterns)")
    print("• Personality Percentages (MBTI-style: I/E, N/S, F/T, P/J)")


if __name__ == "__main__":
    main()