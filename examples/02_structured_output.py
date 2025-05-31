"""Get structured, typed responses from LLMs.

This example shows how to get structured data instead of plain text.
You'll learn:
- How to define output structure with dataclasses
- How to create reusable model bindings
- How to work with typed responses

Requirements:
- ember
- Models: gpt-4 or gpt-3.5-turbo

Expected output:
    City: Paris
    Country: France
    Population: 2161000
    Fun fact: Paris is known as "The City of Light"
"""

from dataclasses import dataclass
from ember.api import models


@dataclass
class CityInfo:
    """Structured information about a city."""
    name: str
    country: str
    population: int
    fun_fact: str


def get_city_info(city_name: str) -> CityInfo:
    """Get structured information about a city."""
    # Create a model instance for reuse
    model = models.instance("gpt-4", temperature=0.3)
    
    # Ask for specific format
    prompt = f"""
    Provide information about {city_name} in this exact format:
    CITY: [name]
    COUNTRY: [country]
    POPULATION: [approximate number]
    FUN_FACT: [one interesting fact]
    """
    
    response = model(prompt)
    
    # Parse the response (in production, use proper parsing)
    lines = response.text.strip().split('\n')
    data = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            data[key.strip()] = value.strip()
    
    return CityInfo(
        name=data.get('CITY', city_name),
        country=data.get('COUNTRY', 'Unknown'),
        population=int(data.get('POPULATION', '0').replace(',', '')),
        fun_fact=data.get('FUN_FACT', 'No fact available')
    )


def main():
    # Get structured data about Paris
    info = get_city_info("Paris")
    
    # Work with typed data
    print(f"City: {info.name}")
    print(f"Country: {info.country}")
    print(f"Population: {info.population:,}")
    print(f"Fun fact: {info.fun_fact}")
    
    # The benefit: IDE autocomplete and type checking work!
    # Try: info.<TAB> in your IDE


if __name__ == "__main__":
    main()


# Next steps:
# - See examples/patterns/structured_extraction.py for advanced parsing
# - Use Pydantic for automatic validation
# - Learn about JSON mode for guaranteed structure