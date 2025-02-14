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

function VisionAgent(img: image, context_text: string, primitives_list_string: string) -> VisionAgentOutput {
    client LlamaVisionGroqClient
    prompt #"
        {{_.role("user")}}

        You are a robot navigating in a home.

        Your goal is to decide what to do right now based on the image and the context.

        You are being provided with what the user most recently said, the task that is currently being executed, and the image you see in front of you.

        If there is no task currently being executed, you can choose to do one next or nothing depending on the context.

        If there is a task currently being executed, do not choose to do another task. Do not stop it either.
        
        This is what you see: {{ img }}.

        {{ context_text }}.

        Consider the guidelines of the tasks available carefully before picking one to do (if any).

        You can only use one of the following tasks: {{ primitives_list_string }}.

        {{ ctx.output_format }}
    "#
} 

