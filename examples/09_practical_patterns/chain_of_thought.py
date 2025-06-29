"""Chain of Thought - Step-by-step reasoning for complex problems.

Learn how to implement chain-of-thought prompting to improve reasoning
capabilities and solve complex problems systematically.

Example:
    >>> from ember.api import models
    >>> prompt = "Let's solve this step by step:\\n" + problem
    >>> response = models("gpt-4", prompt)
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from _shared.example_utils import print_section_header, print_example_output


def example_basic_chain_of_thought():
    """Show basic chain-of-thought pattern."""
    print("\n=== Basic Chain of Thought ===\n")

    problem = "If a train travels 120 miles in 2 hours, then speeds up to travel 180 miles in the next 2 hours, what is its average speed for the entire journey?"

    print("Problem:")
    print(f"  {problem}\n")

    print("Without Chain of Thought:")
    print("  'The average speed is 75 mph.'\n")

    print("With Chain of Thought:")
    print("  Step 1: Calculate total distance")
    print("    First segment: 120 miles")
    print("    Second segment: 180 miles")
    print("    Total distance: 120 + 180 = 300 miles\n")

    print("  Step 2: Calculate total time")
    print("    First segment: 2 hours")
    print("    Second segment: 2 hours")
    print("    Total time: 2 + 2 = 4 hours\n")

    print("  Step 3: Calculate average speed")
    print("    Average speed = Total distance / Total time")
    print("    Average speed = 300 miles / 4 hours = 75 mph\n")

    print("  Answer: The average speed is 75 mph.")


def example_structured_reasoning():
    """Demonstrate structured reasoning templates."""
    print("\n\n=== Structured Reasoning Templates ===\n")

    print("Template 1: Problem Decomposition")
    print("  1. Understand: What is being asked?")
    print("  2. Identify: What information is given?")
    print("  3. Plan: What steps are needed?")
    print("  4. Execute: Work through each step")
    print("  5. Verify: Check the answer\n")

    print("Template 2: Hypothesis Testing")
    print("  1. State hypothesis")
    print("  2. List supporting evidence")
    print("  3. List contradicting evidence")
    print("  4. Evaluate strength of evidence")
    print("  5. Draw conclusion\n")

    print("Template 3: Comparative Analysis")
    print("  1. Identify options")
    print("  2. Define criteria")
    print("  3. Evaluate each option")
    print("  4. Compare results")
    print("  5. Make recommendation")


def example_math_problem_solving():
    """Show chain of thought for math problems."""
    print("\n\n=== Mathematical Problem Solving ===\n")

    problem = "A baker has 24 eggs. Each cake requires 3 eggs and each batch of cookies requires 2 eggs. If the baker makes 5 cakes, how many batches of cookies can be made with the remaining eggs?"

    print("Problem:")
    print(f"  {problem}\n")

    print("Chain of Thought Solution:")
    print("  Given information:")
    print("    • Total eggs: 24")
    print("    • Eggs per cake: 3")
    print("    • Eggs per cookie batch: 2")
    print("    • Number of cakes to make: 5\n")

    print("  Step 1: Calculate eggs used for cakes")
    print("    Eggs for cakes = 5 cakes × 3 eggs/cake = 15 eggs\n")

    print("  Step 2: Calculate remaining eggs")
    print("    Remaining eggs = 24 - 15 = 9 eggs\n")

    print("  Step 3: Calculate cookie batches possible")
    print("    Cookie batches = 9 eggs ÷ 2 eggs/batch = 4.5 batches")
    print("    Since we can't make half a batch: 4 batches\n")

    print("  Answer: 4 batches of cookies")


def example_logical_reasoning():
    """Demonstrate logical reasoning chains."""
    print("\n\n=== Logical Reasoning ===\n")

    print("Logical puzzle:")
    print("  'All roses are flowers. Some flowers fade quickly.")
    print("   No roses are blue. Can we conclude anything about")
    print("   blue flowers fading quickly?'\n")

    print("Chain of Thought:")
    print("  1. Parse the statements:")
    print("     • All roses are flowers (roses ⊆ flowers)")
    print("     • Some flowers fade quickly")
    print("     • No roses are blue (roses ∩ blue = ∅)\n")

    print("  2. What do we know about blue flowers?")
    print("     • Blue flowers exist (implied)")
    print("     • Blue flowers are not roses")
    print("     • Blue flowers are flowers\n")

    print("  3. Can blue flowers fade quickly?")
    print("     • Some flowers fade quickly")
    print("     • Blue flowers are flowers")
    print("     • Therefore, blue flowers COULD fade quickly\n")

    print("  4. Conclusion:")
    print("     We cannot determine if blue flowers fade quickly.")
    print("     The information is insufficient.")


def example_code_debugging():
    """Show chain of thought for debugging."""
    print("\n\n=== Code Debugging with Chain of Thought ===\n")

    print("Bug report: 'Function returns wrong sum for list [1, 2, 3]'")
    print("\nBuggy code:")
    print("  def sum_list(lst):")
    print("      total = 0")
    print("      for i in range(len(lst)):")
    print("          total = lst[i]")
    print("      return total\n")

    print("Chain of Thought Debugging:")
    print("  1. Understand expected behavior:")
    print("     Input: [1, 2, 3]")
    print("     Expected: 6 (1 + 2 + 3)")
    print("     Actual: 3\n")

    print("  2. Trace execution:")
    print("     i=0: total = lst[0] = 1")
    print("     i=1: total = lst[1] = 2")
    print("     i=2: total = lst[2] = 3")
    print("     Return: 3\n")

    print("  3. Identify issue:")
    print("     Using '=' instead of '+='")
    print("     Each iteration overwrites total\n")

    print("  4. Fix:")
    print("     Change 'total = lst[i]' to 'total += lst[i]'")


def example_multi_step_analysis():
    """Demonstrate multi-step analysis."""
    print("\n\n=== Multi-Step Analysis ===\n")

    print("Analyzing a business decision:\n")

    print("Question: Should we launch product in Region A or Region B?")
    print("\nStep 1: Identify factors")
    print("  • Market size")
    print("  • Competition")
    print("  • Regulatory environment")
    print("  • Distribution costs")
    print("  • Cultural fit\n")

    print("Step 2: Gather data")
    print("  Region A:")
    print("    • Market size: $10M")
    print("    • Competitors: 3 major")
    print("    • Regulations: Moderate")
    print("    • Distribution: $$$")
    print("    • Cultural fit: High\n")

    print("  Region B:")
    print("    • Market size: $15M")
    print("    • Competitors: 1 major")
    print("    • Regulations: Light")
    print("    • Distribution: $")
    print("    • Cultural fit: Medium\n")

    print("Step 3: Weighted analysis")
    print("  (Weights: Size=0.3, Competition=0.2, Reg=0.2, Cost=0.2, Fit=0.1)")
    print("  Region A: 0.3(7) + 0.2(4) + 0.2(6) + 0.2(3) + 0.1(9) = 5.6")
    print("  Region B: 0.3(9) + 0.2(8) + 0.2(8) + 0.2(9) + 0.1(6) = 8.3\n")

    print("Step 4: Recommendation")
    print("  Region B scores higher (8.3 vs 5.6)")
    print("  Recommend: Launch in Region B")


def example_error_recovery():
    """Show chain of thought for error recovery."""
    print("\n\n=== Error Recovery Reasoning ===\n")

    print("Scenario: API call failed with timeout\n")

    print("Chain of Thought Recovery:")
    print("  1. Identify error type:")
    print("     • Timeout error")
    print("     • Not a client error (4xx)")
    print("     • Possibly transient\n")

    print("  2. Consider possible causes:")
    print("     • Network congestion")
    print("     • Server overload")
    print("     • Large request size")
    print("     • Rate limiting\n")

    print("  3. Determine recovery strategy:")
    print("     • Check if idempotent → Yes")
    print("     • Check retry budget → 2 retries left")
    print("     • Check backoff needed → Yes\n")

    print("  4. Execute recovery:")
    print("     • Wait 2 seconds (exponential backoff)")
    print("     • Retry with same parameters")
    print("     • If fails again, try with smaller batch")


def example_prompt_engineering_cot():
    """Show prompt engineering for chain of thought."""
    print("\n\n=== Prompt Engineering for CoT ===\n")

    print("Effective CoT prompt patterns:\n")

    print("1. Explicit instruction:")
    print('   "Let\'s solve this step by step:"\n')

    print("2. Few-shot with reasoning:")
    print('   "Q: [Problem 1]"')
    print('   "A: Let me work through this:"')
    print('   "   Step 1: ..."')
    print('   "   Step 2: ..."')
    print('   "   Therefore: ..."\n')

    print("3. Structured format:")
    print('   "Problem: [statement]"')
    print('   "Approach: Break down into steps"')
    print('   "Solution:"')
    print('   "1. First, ..."')
    print('   "2. Then, ..."')
    print('   "3. Finally, ..."\n')

    print("4. Self-consistency:")
    print('   "Solve this problem three different ways"')
    print('   "and verify they give the same answer."')


def main():
    """Run all chain of thought examples."""
    print_section_header("Chain of Thought Reasoning")

    print("🎯 Why Chain of Thought Works:\n")
    print("• Breaks complex problems into steps")
    print("• Reduces errors through explicit reasoning")
    print("• Improves interpretability")
    print("• Enables verification of logic")
    print("• Handles multi-step problems better")

    example_basic_chain_of_thought()
    example_structured_reasoning()
    example_math_problem_solving()
    example_logical_reasoning()
    example_code_debugging()
    example_multi_step_analysis()
    example_error_recovery()
    example_prompt_engineering_cot()

    print("\n" + "=" * 50)
    print("✅ Chain of Thought Best Practices")
    print("=" * 50)
    print("\n1. Be explicit about each step")
    print("2. Show intermediate calculations")
    print("3. State assumptions clearly")
    print("4. Verify logic at each stage")
    print("5. Use consistent formatting")
    print("6. Include sanity checks")
    print("7. Explain the 'why' not just 'what'")

    print("\n🔧 Implementation Tips:")
    print("• Start with 'Let's think step by step'")
    print("• Number or bullet each step")
    print("• Use clear transitions between steps")
    print("• Summarize at the end")
    print("• Check work when possible")

    print("\nNext: Explore evaluation in '../10_evaluation_suite/'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
