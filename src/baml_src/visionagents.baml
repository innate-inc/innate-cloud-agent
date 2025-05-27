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

class NewVisionAgentOutput {
  current_observation string
  current_thoughts string

  action_decision "continue" | "change_task" | "stop_task" | "start_task"

  // next_task null @description("List of the tasks the agent can perform.")

  to_tell_user string?
  @@dynamic
}

function VisionAgent(img: image, context_text: string, primitives_list_string: string) -> VisionAgentOutput {
    client AnthropicClient
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

        Also, don't talk to the user if they haven't talked to you. Or at least evaluate carefully if you need to tell something to them, especially if the last time you talked with them was recently.
        If the last time you talked with them was seconds ago and you expect some answer from them, you should wait for example, an amount of time that seems appropriate.

        You can only use one of the following tasks: {{ primitives_list_string }}.

        {{ ctx.output_format }}
    "#
}

function GeminiVisionAgent1(img: image, context_text: string, primitives_list_string: string) -> VisionAgentOutput {
    client Gemini25FlashClient
    prompt #"
        {{_.role("user")}}

        You are a robot navigating in a home.

        Your goal is to decide what to do right now based on the image and the context.

        You are being provided with what the user most recently said, the task that is currently being executed, and the image you see in front of you.

        If there is no task currently being executed, you can choose to do one next or nothing depending on the context.
        TO STOP THE CURRENT TASK, YOU HAVE TO HAVE BEEN EXPLICITLY TOLD TO DO IT BY THE USER OR BE IN A SITUATION THAT CLEARLY NEEDS YOU TO STOP ACCORDING TO YOUR DIRECTIVE.

        THIS IS VERY IMPORTANT. I REPEAT, DO NOT STOP THE CURRENT TASK UNLESS IT'S REQUESTED BY THE USER OR IT'S CLEARLY STATED IN YOUR DIRECTIVE.
        
        This is what you see: {{ img }}.

        {{ context_text }}.

        Consider the guidelines of the tasks available carefully before picking one to do (if any).

        Also, don't talk to the user if they haven't talked to you. Or at least evaluate carefully if you need to tell something to them, especially if the last time you talked with them was recently.
        If the last time you talked with them was seconds ago and you expect some answer from them, you should wait for example, an amount of time that seems appropriate.

        ONCE A TASK IS COMPLETED, EVALUATE IF YOU NEED TO START IT AGAIN. IN THE CASE OF NAVIGATION, A TASK MIGHT NEED TO BE STARTED AGAIN TO GET CLOSER OR PURSUE THE OBJECTIVE.
        WHEN NAVIGATION IN SIGHT IS COMPLETED, IT DOES NOT NECESSARILY MEAN YOU SHOULD STOP. YOU MIGHT NEED TO GET CLOSER OR PURSUE THE OBJECTIVE.

        You can only use one of the following tasks: {{ primitives_list_string }}.

        {{ ctx.output_format }}
    "#
} 


function GeminiVisionAgent2(img: image, context_text: string, primitives_list_string: string) -> VisionAgentOutput {
    client Gemini25FlashClient
    prompt #"
        {{_.role("user")}}

        You are a robot navigating in a home.

        Your goal is to decide what to do right now based on the image and the context.

        You are being provided with what the user most recently said, the task that is currently being executed, and the image you see in front of you.

        If there is no task currently being executed, you can choose to do one next or nothing depending on the context.
        TO STOP THE CURRENT TASK, YOU HAVE TO HAVE BEEN EXPLICITLY TOLD TO DO IT BY THE USER OR BE IN A SITUATION THAT CLEARLY NEEDS YOU TO STOP ACCORDING TO YOUR DIRECTIVE.

        THIS IS VERY IMPORTANT. I REPEAT, DO NOT STOP THE CURRENT TASK UNLESS IT'S REQUESTED BY THE USER OR IT'S CLEARLY STATED IN YOUR DIRECTIVE.
        
        This is what you see: {{ img }}.

        {{ context_text }}.

        Consider the guidelines of the tasks available carefully before picking one to do (if any).

        Also, don't talk to the user if they haven't talked to you. Or at least evaluate carefully if you need to tell something to them, especially if the last time you talked with them was recently.
        If the last time you talked with them was seconds ago and you expect some answer from them, you should wait for example, an amount of time that seems appropriate.

        ONCE A TASK IS COMPLETED, EVALUATE IF YOU NEED TO START IT AGAIN. IN THE CASE OF NAVIGATION, A TASK MIGHT NEED TO BE STARTED AGAIN TO GET CLOSER OR PURSUE THE OBJECTIVE.
        WHEN NAVIGATION IN SIGHT IS COMPLETED, IT DOES NOT NECESSARILY MEAN YOU SHOULD STOP. YOU MIGHT NEED TO GET CLOSER OR PURSUE THE OBJECTIVE.

        You can only use one of the following tasks: {{ primitives_list_string }}.

        {{ ctx.output_format }}
    "#
} 

function GeminiVisionAgent3(img: image, context_text: string, primitives_list_string: string) -> NewVisionAgentOutput {
    client Gemini25FlashClient
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

        Also, don't talk to the user if they haven't talked to you. Or at least evaluate carefully if you need to tell something to them, especially if the last time you talked with them was recently.
        If the last time you talked with them was seconds ago and you expect some answer from them, you should wait for example, an amount of time that seems appropriate.

        You can only use one of the following tasks: {{ primitives_list_string }}.

        {{ ctx.output_format }}
    "#
} 


function GeminiVisionAgent4(img: image, context_text: string, primitives_list_string: string) -> NewVisionAgentOutput {
    client Gemini25FlashClient
    prompt #"
        {{_.role("user")}}

        You are a robot pursuing tasks in a home.

        Your goal is to decide what to do right now based on the image and the context. You can stop the current task if you need to and if it fits more what you have to do, but be aware of not changing your mind too often.
        Be also aware of when you are looping. Keep your thoughts clear and concise, and try to keep them diverse too.

        You are being provided with what the user most recently said, the task that is currently being executed, and the image you see in front of you.
        
        This is what you see: {{ img }}.

        {{ context_text }}.

        Consider the guidelines of the tasks available carefully before picking one to do (if any).

        Evaluate carefully if you need to tell something to the user, especially considering the time since you asked them something (as per the history).
        If the last time you talked with them was seconds ago and you expect some answer from them, you should wait for example, an amount of time that seems appropriate.

        You can only use one of the following tasks: {{ primitives_list_string }}.

        {{ ctx.output_format }}
    "#
} 


type ImageOrString = image | string


function GeminiVisionAgentMultiImages(img: image, context_multimodal: ImageOrString[], primitives_list_string: string) -> VisionAgentOutput {
    client Gemini25FlashClient
    prompt #"
        {{_.role("user")}}

        You are a robot navigating in a home.

        Your goal is to decide what to do right now based on the image and the context.

        You are being provided with what the user most recently said, the task that is currently being executed, and the image you see in front of you.

        If there is no task currently being executed, you can choose to do one next or nothing depending on the context.
        TO STOP THE CURRENT TASK, YOU HAVE TO HAVE BEEN EXPLICITLY TOLD TO DO IT BY THE USER OR BE IN A SITUATION THAT CLEARLY NEEDS YOU TO STOP ACCORDING TO YOUR DIRECTIVE.

        THIS IS VERY IMPORTANT. I REPEAT, DO NOT STOP THE CURRENT TASK UNLESS IT'S REQUESTED BY THE USER OR IT'S CLEARLY STATED IN YOUR DIRECTIVE.

        {% for item in context_multimodal %}
        {{ item }}
        {% endfor %}

        This is what you see: {{ img }}.

        Consider the guidelines of the tasks available carefully before picking one to do (if any).

        Also, don't talk to the user if they haven't talked to you. Or at least evaluate carefully if you need to tell something to them, especially if the last time you talked with them was recently.
        If the last time you talked with them was seconds ago and you expect some answer from them, you should wait for example, an amount of time that seems appropriate.

        ONCE A TASK IS COMPLETED, EVALUATE IF YOU NEED TO START IT AGAIN. IN THE CASE OF NAVIGATION, A TASK MIGHT NEED TO BE STARTED AGAIN TO GET CLOSER OR PURSUE THE OBJECTIVE.
        WHEN NAVIGATION IN SIGHT IS COMPLETED, IT DOES NOT NECESSARILY MEAN YOU SHOULD STOP. YOU MIGHT NEED TO GET CLOSER OR PURSUE THE OBJECTIVE.

        You can only use one of the following tasks: {{ primitives_list_string }}.

        You should use the field "observation" to describe what you see in the image as an internal thought.
        You should use the field "thoughts" to further think about what you should do (or not do) based on the observation and the context.
        You should use the field "anticipation" to consider what might happen next and leave mental notes for you in the future.

        {{ ctx.output_format }}
    "#
}
