import json
from pydantic import BaseModel, Field
from typing import Optional, List, Iterator
from agno.workflow import Workflow, RunResponse, RunEvent
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.utils.log import logger
from agno.storage.workflow.sqlite import SqliteWorkflowStorage
from agno.playground import Playground, serve_playground_app

class NewsArticle(BaseModel):
    title: str = Field(name='Article Title', description='Title of the article.')
    url: str = Field(name='Article URL', description='Link to the article.')
    summary: Optional[str] = Field(name='Article Summary', description='Summary of the article if available.')

class SearchResults(BaseModel):
    articles: List[NewsArticle]

class BlogPostGenerator(Workflow):
    searcher = Agent(
        model=OpenAIChat(id='gpt-4o-mini'),
        tools=[DuckDuckGoTools()],
        instructions=["Given a topic, search for the top 5 articles."],
        add_datetime_to_instructions=True,
        response_model=SearchResults,
        structured_outputs=True,
        debug_mode=True,
        show_tool_calls=True
    )

    writer = Agent(
        model=Gemini(id='gemini-2.0-flash-thinking-exp-01-21'),
        instructions=[
            "You will be provided with a topic and a list of top articles.",
            "Generate a New York Times-worthy blog post with catchy sections.",
            "Include key takeaways and always cite sources."        
        ],
        debug_mode=True,
        markdown=True,
    )

    def run(self, topic: str, use_cache: bool = True) -> Iterator[RunResponse]:
        logger.info(f"Generating a blog post on: {topic}")
        logger.info(f"use_cache: {use_cache}")

        # Step 1: Check Cache
        if use_cache:
            cached_blog_post = self._get_cached_blog_post(topic)
            if cached_blog_post:
                logger.info(f"Using cached blog post for topic: '{topic}'")
                yield RunResponse(content=cached_blog_post, event=RunEvent.workflow_completed)
                return

        # Step 2: Search for Articles
        search_results = self._get_search_results(topic)
        if not search_results or len(search_results.articles) == 0:
            logger.warning(f"No search results found for topic: '{topic}'")
            yield RunResponse(content=f"No search results found for topic: '{topic}'", event=RunEvent.workflow_completed)
            return

        # Step 3: Write the Blog Post
        yield from self._write_blog_post(topic, search_results)

    def _add_blog_post_to_cache(self, topic: str, blog_post: Optional[str]) -> None:
        logger.info(f"Caching blog post for topic: '{topic}'")

        self.session_state.setdefault('blog_posts', {})
        self.session_state['blog_posts'][topic] = blog_post

        logger.info(f"Blog post cached successfully for topic: '{topic}'")

    def _get_cached_blog_post(self, topic: str) -> Optional[str]:
        logger.info(f"Checking cache for blog post on topic: '{topic}'")

        cached_post = self.session_state.get('blog_posts', {}).get(topic)

        if cached_post:
            logger.info(f"Cache hit for topic: '{topic}'")
        else:
            logger.info(f"No cached blog post found for topic: '{topic}'")

        return cached_post
    
    def _get_search_results(self, topic: str) -> Optional[SearchResults]:
        MAX_ATTEMPTS = 3
        for attempt in range(MAX_ATTEMPTS):
            try:
                logger.info(f"Attempt {attempt + 1}: Searching for articles on '{topic}'")
                response = self.searcher.run(topic)

                if response and isinstance(response.content, SearchResults):
                    logger.info(f"Found {len(response.content.articles)} articles on attempt {attempt + 1}")
                    return response.content

                logger.warning(f"Attempt {attempt + 1}: Invalid or empty response")
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed with error: {e}")

        logger.error(f"Failed to retrieve search results for '{topic}' after {MAX_ATTEMPTS} attempts")
        return None
        
    def _write_blog_post(self, topic: str, search_results: SearchResults) -> Iterator[RunResponse]:
        logger.info(f"Writing blog post for topic: '{topic}'")

        writer_input = {
            'topic': topic,
            'articles': [article.model_dump() for article in search_results.articles]
        }

        logger.info(f"Input prepared for writer agent: {json.dumps(writer_input, indent=4)}")

        yield from self.writer.run(json.dumps(writer_input, indent=4), stream=True)

        self._add_blog_post_to_cache(topic, self.writer.run_response.content)


generate_blog_post  = BlogPostGenerator(
    name='Blog Post Generator',
    workflow_id=f'generate_blog_post',
    storage=SqliteWorkflowStorage(
        table_name='generate_blow_post_workflows',
        db_file='./storage/workflows.db'
    )
)

app = Playground(
    workflows=[
        generate_blog_post
    ]
).get_app()


if __name__ == '__main__':
    serve_playground_app('blog_post_generator_workflow_playground:app', reload=True)