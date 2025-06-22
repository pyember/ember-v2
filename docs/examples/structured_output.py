"""
Example: Structured Output
Description: Using Pydantic models for type-safe LLM outputs
Concepts: Structured output, type validation, error handling
"""

from ember.api import ember
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import asyncio


# Define output structures
class ProductReview(BaseModel):
    product_name: str
    rating: float = Field(..., ge=1.0, le=5.0, description="Rating from 1-5")
    pros: List[str] = Field(..., min_items=1, description="Positive aspects")
    cons: List[str] = Field(..., min_items=1, description="Negative aspects")
    summary: str = Field(..., max_length=200)
    recommended: bool


class MovieAnalysis(BaseModel):
    title: str
    genre: List[str]
    themes: List[str]
    mood: str = Field(..., description="Overall mood: uplifting, dark, neutral, etc.")
    target_audience: str
    summary: str
    rating: str = Field(..., pattern="^(G|PG|PG-13|R|NC-17|Not Rated)$")


class TodoItem(BaseModel):
    id: int
    task: str
    priority: str = Field(..., pattern="^(low|medium|high|critical)$")
    due_date: Optional[str] = None
    estimated_hours: float = Field(..., gt=0)
    tags: List[str] = []


async def main():
    # Example 1: Product Review Analysis
    print("=== Product Review Analysis ===")
    
    @ember.op
    async def analyze_review(review_text: str) -> ProductReview:
        """Extract structured information from a product review."""
        prompt = f"""Analyze this product review and extract structured information:

Review: {review_text}

Extract the product name, rating (1-5), pros, cons, summary, and whether it's recommended."""
        
        return await ember.llm(prompt, output_type=ProductReview)
    
    review = """
    I recently bought the TechPro X1 Wireless Headphones and I have mixed feelings. 
    The sound quality is absolutely incredible - crisp highs and deep bass. The 
    battery life is also impressive, lasting over 30 hours. However, they're quite 
    expensive at $299, and the comfort isn't great for long sessions. The ear cups 
    press too hard. Overall, they're good but not perfect. I'd give them 3.5 stars.
    """
    
    try:
        analysis = await analyze_review(review)
        print(f"Product: {analysis.product_name}")
        print(f"Rating: {analysis.rating}/5.0")
        print(f"Pros: {', '.join(analysis.pros)}")
        print(f"Cons: {', '.join(analysis.cons)}")
        print(f"Summary: {analysis.summary}")
        print(f"Recommended: {'Yes' if analysis.recommended else 'No'}")
    except Exception as e:
        print(f"Error analyzing review: {e}")
    
    print()

    # Example 2: Movie Analysis
    print("=== Movie Analysis ===")
    
    movie_description = """
    'Eternal Sunshine of the Spotless Mind' follows Joel and Clementine, former 
    lovers who undergo a procedure to erase memories of each other. As Joel's 
    memories fade, he realizes he wants to preserve them. The film explores themes 
    of memory, identity, and whether painful experiences are worth keeping. It's a 
    melancholic yet hopeful meditation on love and loss, with surreal sequences 
    representing the memory erasure process.
    """
    
    movie_analysis = await ember.llm(
        f"Analyze this movie description: {movie_description}",
        output_type=MovieAnalysis
    )
    
    print(f"Title: {movie_analysis.title}")
    print(f"Genres: {', '.join(movie_analysis.genre)}")
    print(f"Themes: {', '.join(movie_analysis.themes)}")
    print(f"Mood: {movie_analysis.mood}")
    print(f"Target Audience: {movie_analysis.target_audience}")
    print(f"Rating: {movie_analysis.rating}")
    print()

    # Example 3: Generating Structured Data
    print("=== Generating Todo List ===")
    
    @ember.op
    async def create_project_todos(project_description: str) -> List[TodoItem]:
        """Generate a todo list for a project."""
        prompt = f"""Create a detailed todo list for this project:

Project: {project_description}

Generate 5-7 specific tasks with priorities, time estimates, and relevant tags."""
        
        return await ember.llm(prompt, output_type=List[TodoItem])
    
    project = "Build a personal portfolio website with blog functionality"
    todos = await create_project_todos(project)
    
    print(f"Generated {len(todos)} tasks for: {project}")
    for todo in todos:
        print(f"\n[{todo.priority.upper()}] {todo.task}")
        print(f"  Estimated: {todo.estimated_hours} hours")
        if todo.tags:
            print(f"  Tags: {', '.join(todo.tags)}")
        if todo.due_date:
            print(f"  Due: {todo.due_date}")
    
    print()

    # Example 4: Nested Structures
    print("=== Nested Structure Example ===")
    
    class Character(BaseModel):
        name: str
        age: int
        occupation: str
        personality_traits: List[str]
        backstory: str
    
    class PlotPoint(BaseModel):
        chapter: int
        description: str
        characters_involved: List[str]
    
    class StoryOutline(BaseModel):
        title: str
        genre: str
        setting: str
        main_characters: List[Character]
        plot_points: List[PlotPoint]
        themes: List[str]
    
    story_prompt = "A mystery novel set in a 1920s speakeasy"
    
    outline = await ember.llm(
        f"Create a detailed story outline for: {story_prompt}",
        output_type=StoryOutline
    )
    
    print(f"Story: {outline.title}")
    print(f"Genre: {outline.genre}")
    print(f"Setting: {outline.setting}")
    print(f"\nMain Characters:")
    for char in outline.main_characters:
        print(f"  - {char.name} ({char.age}): {char.occupation}")
        print(f"    Traits: {', '.join(char.personality_traits)}")
    
    print(f"\nPlot Points:")
    for point in outline.plot_points[:3]:  # Show first 3
        print(f"  Chapter {point.chapter}: {point.description}")
    
    # Example 5: Error Handling
    print("\n=== Error Handling ===")
    
    class StrictFormat(BaseModel):
        exact_number: float = Field(..., ge=10.0, le=20.0)
        specific_choice: str = Field(..., pattern="^(option_a|option_b|option_c)$")
    
    @ember.op
    async def parse_with_retry(text: str, max_attempts: int = 3) -> Optional[StrictFormat]:
        """Parse text with retry on validation errors."""
        for attempt in range(max_attempts):
            try:
                result = await ember.llm(
                    f"Extract data from: {text}. Number must be between 10-20, "
                    f"choice must be option_a, option_b, or option_c",
                    output_type=StrictFormat
                )
                print(f"Success on attempt {attempt + 1}")
                return result
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    print("Retrying with clearer prompt...")
                    text = f"BE VERY CAREFUL: {text}"
        return None
    
    result = await parse_with_retry("I choose 15.5 and option_b")
    if result:
        print(f"Parsed successfully: number={result.exact_number}, choice={result.specific_choice}")


if __name__ == "__main__":
    print("Ember Structured Output Example")
    print("=" * 50)
    asyncio.run(main())