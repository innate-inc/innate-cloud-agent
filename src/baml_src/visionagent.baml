class VisionAgentOutput {
  stop_current_task bool
  observation string
  thoughts string
  new_goal string?
  
  // Define at runtime next_task as an optional union of the two primitives.
  // next_task null @description("List of the tasks the agent can perform.")
  
  anticipation string?
  to_tell_user string?
  @@dynamic
}

function VisionAgent(img: image, last_user_message: string, primitives_list_string: string) -> VisionAgentOutput {
    client LlamaVisionGroqClient
    prompt #"
        {{_.role("user")}}
        
        Decide what to do next with this image: {{ img }}.

        The last user message was: {{ last_user_message }}.

        Consider the guidelines of the tasks available carefully before picking one to do (if any).

        You can only use one of the following tasks: {{ primitives_list_string }}.

        {{ ctx.output_format }}
    "#
} 

