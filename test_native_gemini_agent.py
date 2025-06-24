#!/usr/bin/env python3
"""
Simple test script to validate the native Gemini vision agent implementation.
This version avoids BAML dependencies.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_schema_generation():
    """
    Test the dynamic schema generation.
    """
    print("🧪 Testing Schema Generation...")

    try:
        from src.agents.types import PrimitiveDefinition
        from src.agents.native_gemini_schema_builder import (
            create_gemini_schema,
            create_response_model,
        )

        # Create sample primitives
        primitives = [
            PrimitiveDefinition(
                name="move_forward",
                guidelines="Move the robot forward by a specified distance",
                inputs={"distance": "float"},
            ),
            PrimitiveDefinition(
                name="turn_left",
                guidelines="Turn the robot left by a specified angle",
                inputs={"angle": "float"},
            ),
            PrimitiveDefinition(
                name="say_hello", guidelines="Make the robot say hello", inputs={}
            ),
        ]

        # Test schema generation (now returns a Pydantic model class)
        schema_model = create_gemini_schema(primitives)
        print("✅ Schema generation succeeded!")
        print(f"   Schema model class: {schema_model.__name__}")

        # Check if the model has the expected fields
        fields = (
            schema_model.model_fields if hasattr(schema_model, "model_fields") else {}
        )
        print(f"   Model has {len(fields)} top-level fields")

        # Test model generation
        model_class = create_response_model(primitives)
        print("✅ Dynamic model generation succeeded!")
        print(f"   Model class: {model_class.__name__}")

        return True

    except Exception as e:
        print(f"❌ Schema generation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_output_conversion():
    """
    Test the output format conversion.
    """
    print("🧪 Testing Output Conversion...")

    try:
        from src.agents.native_gemini_vision_agent import (
            NativeVisionAgentOutput,
            convert_to_legacy_output,
        )

        # Create test native output
        native_output = NativeVisionAgentOutput(
            current_observation="I can see a kitchen in front of me",
            current_thoughts="I should move towards the kitchen",
            action_decision="start_task",
            to_tell_user="I found the kitchen!",
            next_task=None,
        )

        # Convert to legacy format
        legacy_output = convert_to_legacy_output(native_output)

        print("✅ Output conversion succeeded!")
        print(f"   Legacy observation: {legacy_output.observation}")
        print(f"   Legacy thoughts: {legacy_output.thoughts}")
        print(f"   Legacy stop_task: {legacy_output.stop_current_task}")
        print(f"   Legacy to_tell_user: {legacy_output.to_tell_user}")

        return True

    except Exception as e:
        print(f"❌ Output conversion test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_agent_initialization():
    """
    Test the agent initialization without making API calls.
    """
    print("🧪 Testing Agent Initialization...")

    try:
        from src.agents.native_gemini_vision_agent import NativeGeminiVisionAgent

        print("✅ Google Generative AI library is available!")

        # Test initialization with missing API key
        old_key = os.environ.get("GEMINI_API_KEY")
        if old_key:
            del os.environ["GEMINI_API_KEY"]

        try:
            agent = NativeGeminiVisionAgent()
            print("❌ Agent should have raised an error for missing API key")
            return False
        except ValueError as e:
            if "GEMINI_API_KEY environment variable is required" in str(e):
                print("✅ Agent correctly raises error for missing API key!")
            else:
                print(f"❌ Agent raised unexpected error: {e}")
                return False
        finally:
            # Restore the API key if it existed
            if old_key:
                os.environ["GEMINI_API_KEY"] = old_key

        return True

    except Exception as e:
        print(f"❌ Agent initialization test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_native_agent_direct():
    """
    Test the native agent directly if API key is available.
    """
    print("🧪 Testing Native Gemini Agent (Direct)...")

    # Check if API key is available
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not found. Skipping direct API test.")
        return False

    try:
        from src.agents.native_gemini_vision_agent import (
            native_gemini_vision_agent_multimodal_history,
        )
        from src.agents.types import (
            MultimodalVisionAgentInput,
            PrimitiveDefinition,
            MultimodalHistoryItem,
        )

        # Create sample base64 image (1x1 pixel PNG)
        sample_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        # Create sample primitives
        primitives = [
            PrimitiveDefinition(
                name="move_forward",
                guidelines="Move the robot forward by a specified distance",
                inputs={"distance": "float"},
            ),
            PrimitiveDefinition(
                name="turn_left",
                guidelines="Turn the robot left by a specified angle",
                inputs={"angle": "float"},
            ),
        ]

        # Create test inputs
        vlm_inputs = MultimodalVisionAgentInput(
            base64_img=sample_image,
            user_prompt_text="Please help me navigate to the kitchen",
            primitive_in_execution=None,
            primitives_list=primitives,
            multimodal_history=[
                MultimodalHistoryItem(type="text", content="Starting navigation task")
            ],
            robot_coords={"x": 1.0, "y": 2.0, "z": 0.0, "theta": 0.5},
            directive="Navigate safely and efficiently",
            additional_image_data=None,
        )

        # Call the native agent
        result = await native_gemini_vision_agent_multimodal_history(vlm_inputs)

        if result:
            print("✅ Native agent call succeeded!")
            print(f"   Observation: {result.observation[:100]}...")
            print(f"   Thoughts: {result.thoughts[:100]}...")
            print(f"   Stop task: {result.stop_current_task}")
            print(f"   Next task: {result.next_task}")
            return True
        else:
            print("❌ Native agent returned None")
            return False

    except Exception as e:
        print(f"❌ Native agent test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """
    Run all tests.
    """
    print("🚀 Starting Native Gemini Agent Tests (Simplified)")
    print("=" * 60)

    results = []

    # Test schema generation (no dependencies needed)
    results.append(test_schema_generation())

    # Test output conversion (no dependencies needed)
    results.append(test_output_conversion())

    # Test agent initialization (no API calls)
    results.append(test_agent_initialization())

    print("\n" + "=" * 60)
    print("📊 Test Results:")

    passed = sum(1 for r in results if r)
    total = len(results)

    if passed == total:
        print(f"✅ All {total} basic tests passed!")
    else:
        print(f"❌ {passed}/{total} basic tests passed")

    # Try API test if conditions are met
    if os.getenv("GEMINI_API_KEY"):
        print("\n🔑 API key found. Testing direct API call...")
        try:
            api_result = asyncio.run(test_native_agent_direct())
            if api_result:
                print("✅ API test passed!")
                passed += 1
            else:
                print("❌ API test failed")
            total += 1
        except Exception as e:
            print(f"❌ API test error: {e}")
            total += 1
    else:
        print("\n💡 To run API tests, set your GEMINI_API_KEY environment variable")

    print(f"\n🏁 Final Score: {passed}/{total} tests passed")
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
