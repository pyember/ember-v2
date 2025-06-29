"""Template for conditional LLM execution in examples.

This module provides utilities and patterns for creating examples that work
both with and without API keys.
"""

from typing import Dict, Any, List, Optional
from .conditional_execution import SimulatedResponse


def create_simulated_response(
    prompt: str, model: str = "gpt-3.5-turbo"
) -> SimulatedResponse:
    """Create a simulated response based on the prompt content.

    This function analyzes the prompt and returns realistic simulated responses
    that demonstrate the expected behavior without making actual API calls.
    """
    prompt_lower = prompt.lower()

    # Extract entities pattern
    if "extract entities" in prompt_lower:
        return SimulatedResponse(
            "Person: AI researcher, Organization: OpenAI, Location: San Francisco",
            model,
        )

    # Summarize in 10 words pattern
    elif "summarize in 10 words" in prompt_lower:
        return SimulatedResponse(
            "AI transforms work and life through intelligent automation systems", model
        )

    # Extract keywords pattern
    elif "extract 3 keywords" in prompt_lower or "extract keywords" in prompt_lower:
        return SimulatedResponse(
            "artificial intelligence, transformation, technology", model
        )

    # List 2 key points pattern
    elif "list 2 key points" in prompt_lower:
        return SimulatedResponse(
            "1. Main concept explained\n2. Supporting evidence provided", model
        )

    # Summarize sections
    elif "summarize:" in prompt_lower:
        return SimulatedResponse(
            "Section provides comprehensive overview of the topic discussed.", model
        )

    # Extract themes pattern
    elif "identify themes" in prompt_lower or "extract themes" in prompt_lower:
        return SimulatedResponse(
            "Innovation, Technology Advancement, Future Possibilities", model
        )

    # Classification for routing
    elif "classify request type" in prompt_lower:
        if "derivative" in prompt_lower or "math" in prompt_lower:
            return SimulatedResponse("technical", model)
        elif "reverse" in prompt_lower or "function" in prompt_lower:
            return SimulatedResponse("coding", model)
        elif "haiku" in prompt_lower or "poem" in prompt_lower:
            return SimulatedResponse("creative", model)
        else:
            return SimulatedResponse("general", model)

    # Sentiment analysis patterns
    elif "sentiment" in prompt_lower:
        if any(
            word in prompt_lower
            for word in ["love", "amazing", "excellent", "wonderful"]
        ):
            return SimulatedResponse("positive", model)
        elif any(
            word in prompt_lower for word in ["hate", "terrible", "awful", "horrible"]
        ):
            return SimulatedResponse("negative", model)
        else:
            return SimulatedResponse("neutral", model)

    # Emotion detection patterns
    elif "emotion" in prompt_lower:
        if any(word in prompt_lower for word in ["love", "amazing", "wonderful"]):
            return SimulatedResponse("joy", model)
        elif any(word in prompt_lower for word in ["hate", "angry", "furious"]):
            return SimulatedResponse("anger", model)
        elif any(word in prompt_lower for word in ["sad", "terrible", "awful"]):
            return SimulatedResponse("sadness", model)
        else:
            return SimulatedResponse("surprise", model)

    # Intensity scoring patterns
    elif "intensity" in prompt_lower:
        if any(word in prompt_lower for word in ["absolutely", "extremely", "very"]):
            return SimulatedResponse("8", model)
        elif any(word in prompt_lower for word in ["somewhat", "fairly", "quite"]):
            return SimulatedResponse("5", model)
        else:
            return SimulatedResponse("3", model)

    # Topic classification patterns
    elif "topic" in prompt_lower or (
        "classify" in prompt_lower and "topic" in prompt_lower
    ):
        # Extract the text to classify from the prompt
        text_part = prompt_lower
        if ":" in prompt_lower:
            text_part = prompt_lower.split(":", 1)[1]

        if any(
            word in text_part for word in ["iphone", "ai", "technology", "software"]
        ):
            return SimulatedResponse("tech", model)
        elif any(
            word in text_part
            for word in ["team", "championship", "game", "player", "won"]
        ):
            return SimulatedResponse("sports", model)
        elif any(
            word in text_part for word in ["congress", "bill", "policy", "government"]
        ):
            return SimulatedResponse("politics", model)
        elif any(
            word in text_part for word in ["movie", "film", "actor", "box office"]
        ):
            return SimulatedResponse("entertainment", model)
        elif any(
            word in text_part
            for word in ["scientist", "discover", "planet", "exoplanet"]
        ):
            return SimulatedResponse("other", model)
        else:
            return SimulatedResponse("other", model)

    # Translation patterns
    elif "translate" in prompt_lower:
        if "spanish" in prompt_lower:
            translations = {
                "hello, how are you?": "Hola, ¿cómo estás?",
                "thank you very much": "Muchas gracias",
                "good morning": "Buenos días",
            }
            for eng, esp in translations.items():
                if eng in prompt_lower:
                    return SimulatedResponse(esp, model)
            return SimulatedResponse("Buenos días", model)
        elif "french" in prompt_lower:
            if "hello world" in prompt_lower:
                return SimulatedResponse("Bonjour le monde", model)
            return SimulatedResponse("Texte traduit en français", model)

    # Summarization patterns
    elif "summarize" in prompt_lower:
        if "ai research" in prompt_lower:
            return SimulatedResponse(
                "AI research advances natural language understanding capabilities.",
                model,
            )
        elif "stock market" in prompt_lower:
            return SimulatedResponse(
                "Tech earnings drive stock market to record highs.", model
            )
        elif "study" in prompt_lower and "exercise" in prompt_lower:
            return SimulatedResponse(
                "Regular exercise significantly improves mental health outcomes.", model
            )
        elif "climate change" in prompt_lower:
            return SimulatedResponse(
                "Climate change continues to impact global weather patterns.", model
            )
        elif "ai breakthrough" in prompt_lower:
            return SimulatedResponse(
                "Major AI breakthrough enhances language understanding.", model
            )
        elif "medical" in prompt_lower or "treatment" in prompt_lower:
            return SimulatedResponse(
                "Medical breakthrough offers new treatment possibilities.", model
            )
        else:
            return SimulatedResponse(
                "This text discusses important developments in its field.", model
            )

    # Key points extraction
    elif "key points" in prompt_lower or "list 3" in prompt_lower:
        return SimulatedResponse(
            "1. Main concept identified\n2. Supporting evidence found\n3. Conclusion reached",
            model,
        )

    # Category detection
    elif "category" in prompt_lower:
        if "research" in prompt_lower or "study" in prompt_lower:
            return SimulatedResponse("research", model)
        elif (
            "news" in prompt_lower
            or "market" in prompt_lower
            or "stock" in prompt_lower
        ):
            return SimulatedResponse("news", model)
        else:
            return SimulatedResponse("blog", model)

    # Analysis patterns
    elif "analyze" in prompt_lower:
        if "english" in prompt_lower:
            return SimulatedResponse(
                "This English text contains common words and standard grammar.", model
            )
        elif "spanish" in prompt_lower:
            return SimulatedResponse(
                "Este texto en español contiene palabras comunes y gramática estándar.",
                model,
            )
        elif "french" in prompt_lower:
            return SimulatedResponse(
                "Ce texte français contient des mots courants et une grammaire standard.",
                model,
            )
        elif "german" in prompt_lower:
            return SimulatedResponse(
                "Dieser deutsche Text enthält häufige Wörter und Standardgrammatik.",
                model,
            )
        else:
            return SimulatedResponse("Text analysis completed successfully.", model)

    # Technical/math queries
    elif "derivative" in prompt_lower:
        return SimulatedResponse("The derivative of x^2 + 3x is 2x + 3", model)

    # Coding queries
    elif "reverse a string" in prompt_lower:
        return SimulatedResponse("def reverse_string(s): return s[::-1]", model)

    # Creative queries
    elif "haiku" in prompt_lower:
        return SimulatedResponse(
            "Spring blossoms bloom bright\nNature awakens from sleep\nLife begins anew",
            model,
        )

    # General knowledge queries
    elif "capital of japan" in prompt_lower:
        return SimulatedResponse("The capital of Japan is Tokyo", model)

    # Meta-programming patterns
    elif "generate validator for" in prompt_lower:
        if "age" in prompt_lower:
            return SimulatedResponse(
                "def validate(value): return isinstance(value, int) and 0 <= value <= 150",
                model,
            )
        elif "email" in prompt_lower:
            return SimulatedResponse(
                "def validate(value): return isinstance(value, str) and '@' in value",
                model,
            )
        else:
            return SimulatedResponse(
                "def validate(value): return value is not None", model
            )

    # Default response
    else:
        return SimulatedResponse(f"Processed: {prompt[:50]}...", model)


def simulated_models(model_id: str, prompt: str, **kwargs) -> SimulatedResponse:
    """Simulated version of the models() function for examples.

    This function returns realistic responses without making actual API calls,
    allowing examples to demonstrate behavior without requiring API keys.
    """
    return create_simulated_response(prompt, model_id)


def get_simulated_batch_results(
    prompts: List[str], model: str = "gpt-3.5-turbo"
) -> List[SimulatedResponse]:
    """Get simulated results for batch processing examples."""
    return [create_simulated_response(prompt, model) for prompt in prompts]
