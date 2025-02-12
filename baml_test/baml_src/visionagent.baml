// Define the input structure for ServeGlassPrimitive.
class ServeGlassPrimitiveInput {
  // The specific input key "distance" of type float.
  distance float
}

// Define the ServeGlassPrimitive composite type.
class ServeGlassPrimitive {
  // The guideline for using this primitive.
  guideline string @description("Serve a glass of water. The glass has to be in sight.")
  
  // The composite input structure specific to this primitive.
  inputs ServeGlassPrimitiveInput
}

// Define the input structure for GrabBottlePrimitive.
class GrabBottlePrimitiveInput {
  // For example, specify the color of the bottle.
  bottle_color string
}

// Define the GrabBottlePrimitive composite type.
class GrabBottlePrimitive {
  // The guideline for this primitive.
  guideline string @description("Grab a bottle with a specified color. The bottle must be within reach.")
  
  // The composite input structure specific to this primitive.
  inputs GrabBottlePrimitiveInput
}

class VisionAgentOutput {
  stop_current_task bool
  observation string
  thoughts string
  new_goal string?
  
  // Define next_task as an optional union of the two primitives.
  next_task (ServeGlassPrimitive | GrabBottlePrimitive)? @description("Optional next task as a union of primitives")
  
  anticipation string?
  to_tell_user string?
}

function VisionAgent(img: image) -> VisionAgentOutput {
    client OpenAIClient
    prompt #"
        {{_.role("user")}}
        
        Decide what to do next with this image: {{ img }}.

        Consider the guidelines of the primitives carefully before picking one to do (if any).

        {{ ctx.output_format }}
    "#
} 

