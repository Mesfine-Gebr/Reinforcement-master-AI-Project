# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        

        for _ in range(self.iterations):
            new_values = self.values.copy()
            

            # get Q_values for each possible s_prime
            for state in self.mdp.getStates():
                Q_values = [float('-inf')]
                terminal_state = self.mdp.isTerminal(state)  # boolean

                # Terminal states have 0 value.
                if terminal_state:
                    new_values[state] = 0

                else:
                    legal_actions = self.mdp.getPossibleActions(state)

                    for action in legal_actions:
                        Q_values.append(self.getQValue(state, action))

                    # update value function at state s to largest Q_value
                    new_values[state] = max(Q_values)

            self.values = new_values



        




    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]



    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        
        # Get the possible next states and their corresponding transition probabilities
        next_states = self.mdp.getTransitionStatesAndProbs(state, action)
        Ut = []

         # Iterate over each next state and its corresponding transition probability
        for state_prime, Trans in next_states:

             # Get the reward for the current state-action-state_prime transition
            reward = self.mdp.getReward(state, action, state_prime)

            # Calculate the discounted value of the next state
            gamma_V = self.discount * self.values[state_prime]
            Ut.append(Trans * (reward + gamma_V))

        return sum(Ut)

        

        
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Get the list of possible actions in the given state
        actions = self.mdp.getPossibleActions(state)

        # If there are no legal actions (terminal state), return None
        if len(actions) == 0:
            return None

        actions_qvals = []  
        # Iterate over each action and calculate its Q-value
        for action in actions:
            # Get the Q-value for the current state-action pair
            qval = self.getQValue(state, action)
            
            # Append the action-value pair to the list
            actions_qvals.append((action, qval))
            
        # Find the action with the highest Q-value
        best_qval = max(actions_qvals, key=lambda x: x[1])[0]
        return best_qval

        
        util.raiseNotDefined()

        

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # Get all the states in the MDP
        states = self.mdp.getStates()

        # initialize value function to all 0 values.
        for state in states:
            self.values[state] = 0
            
        # Calculate the total number of states
        number_of_states = len(states)
        
        # Perform value iteration for the specified number of iterations
        for s in range(self.iterations):
            # Determine the current state based on the iteration index
            state_index = s % number_of_states
            state = states[state_index]

            # Check if the current state is a terminal state
            terminal = self.mdp.isTerminal(state)
            if not terminal:
                iteration_action = self.getAction(state)
                
                # Calculate the Q-value for the current state-action pai
                qval = self.getQValue(state, iteration_action)
                # Update the value function with the calculated Q-value
                self.values[state] = qval


        

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Get all the states in the MDP
        states = self.mdp.getStates()
        
        # Create a priority queue for fringe states
        fringe = util.PriorityQueue()
        
        # Create a dictionary to store predecessors for each state
        predecessors = {}

        # Initialize the value function and predecessors
        for state in states:
            self.values[state] = 0
            predecessors[state] = self.get_predecessors(state)

        # Populate the fringe with non-terminal states based on value differences
        for state in states:
            terminal = self.mdp.isTerminal(state)

            if not terminal:
                current_value_of_state = self.values[state]
                difference = abs(current_value_of_state - self.max_Qvalue(state))
                fringe.push(state, -difference)

        # Perform value iteration for the specified number of iterations
        for _ in range(self.iterations):

            # Check if the fringe is empty
            if fringe.isEmpty():
                return
            
            # Get the state with the highest value difference from the fringe
            state = fringe.pop()

            # Update the value of the current state with the maximum Q-value
            self.values[state] = self.max_Qvalue(state)

            # Update the fringe for the predecessors of the current state
            for predecessor in predecessors[state]:
                difference = abs(self.values[predecessor] - self.max_Qvalue(predecessor))
                if difference > self.theta:
                    fringe.update(predecessor, -difference)


    def max_Qvalue(self, state):
        return max([self.getQValue(state, a) for a in self.mdp.getPossibleActions(state)])


    # First, define the predecessors of a state s as all states that have
    # a nonzero probability of reaching s by taking some action a
    # This means no Terminal states and T > 0.
    def get_predecessors(self, state):
        predecessor_set = set()
        states = self.mdp.getStates()
        action_direction = ['north', 'south', 'east', 'west']

        # Check if the given state is not a terminal state
        if not self.mdp.isTerminal(state):

            # Iterate over all the states in the MDP
            for predecessor in states:
                terminal = self.mdp.isTerminal(predecessor)
                legal_actions = self.mdp.getPossibleActions(predecessor)

                 # Check if the predecessor state is not a terminal state
                if not terminal:

                    # Iterate over the possible actions in action_direction list
                    for act in action_direction:

                        # Check if the action is a legal action for the predecessor state
                        if act in legal_actions:
                            transition = self.mdp.getTransitionStatesAndProbs(predecessor, act)

                             # Iterate over the transition states
                            for state_prime, Trans in transition:

                                # Check if the transition state is equal to the given state and has a non-zero probability
                                if (state_prime == state) and (Trans > 0):
                                    predecessor_set.add(predecessor)

        return predecessor_set



        

