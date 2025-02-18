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
    client OpenAIClient
    prompt #"
        {{_.role("user")}}

        You are a robot navigating in a home.

        Your goal is to decide what to do right now based on the image and the context.

        You are being provided with what the user most recently said, the task that is currently being executed, and the image you see in front of you.

        If there is no task currently being executed, you can choose to do one next or nothing depending on the context.
        TO SET A NEW TASK, YOU HAVE TO HAVE BEEN EXPLICITLY TOLD TO DO IT BY THE USER. IF THE USER DID NOT SAY ANYTHING, YOU SHOULD NOT SET A NEW TASK.

        If there is a task currently being executed, do not choose to do another task. Do not stop it either. 
        THIS IS VERY IMPORTANT. I REPEAT, DO NOT STOP THE CURRENT TASK IF YOU ARE EXECUTING ONE.
        
        This is what you see: {{ img }}.

        {{ context_text }}.

        Consider the guidelines of the tasks available carefully before picking one to do (if any).

        You can only use one of the following tasks: {{ primitives_list_string }}.

        {{ ctx.output_format }}
    "#
} 

