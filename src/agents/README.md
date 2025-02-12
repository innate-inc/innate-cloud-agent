**Core Agent Method Overview:**

- **Purpose:**
  - Execute control decisions at arbitrarily fast speeds based on the robot's current state and sensory history.

- **Inputs:**
  - **Current State of the Robot:**  
    - The immediate status information of the robot.
  - **History of Sensory Inputs:**  
    - A record of the recent sensory data that the robot has received.
  - **Directive:**
    - The directive is the system prompt of the agent, defined by the user
  - **Primitives:**
    - The primitives are the actions that the agent can perform, defined by the user.

- **Output:**
  - **Delta for Actions:**  
    - A structured object specifying:
      - What to say.
      - What to do.
      - Whether to stop.

- **Architecture Flexibility:**
  - Different core agents can be defined as long as they respect these inputs and outputs.
  - This allows experimentation with agents that operate:
    - Faster.
    - Slower.
    - In multiple steps. 