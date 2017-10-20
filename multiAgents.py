# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        #print "Successor Game State: ", successorGameState
        #print "new position: ", newPos
        #print "remaining food: ", newFood
        #print "new ghost states: ", newGhostStates
        #print "scared timer: ", newScaredTimes

        evalNum = 0.0
        currentFood = currentGameState.getFood()

        for food in currentFood.asList():
            foodDist = manhattanDistance(newPos, food)
            if foodDist == 0:
                evalNum += 2
            else:
                evalNum += 1.0 / foodDist

        for ghost in newGhostStates:
          ghostDist = manhattanDistance(newPos, ghost.getPosition())
          if ghostDist <= 3:
              evalNum -= 3

        return evalNum

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        return self.maxValue(gameState, self.depth)[1]

    def maxValue(self, gameState, depth):
        # if terminal test then return utility(state)
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), ""
        # value = -infinity
        scores = []
        # for loop through actions(state) and find max value
        actions = gameState.getLegalActions()
        for a in actions:
            scores.append(self.minValue(gameState.generateSuccessor(self.index, a), depth, 1))
        maxScore = max(scores)
        act = scores.index(maxScore)

        return maxScore, actions[act]

    def minValue(self, gameState, depth, agentIndex):
        # if terminal test then return utility(state)
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), ""
        # value = -infinity
        scores = []
        # for loop through actions(state) and find min value
        actions = gameState.getLegalActions(agentIndex)
        for a in actions:
            if(agentIndex == gameState.getNumAgents() - 1):
                scores.append(self.maxValue(gameState.generateSuccessor(agentIndex, a), (depth-1))[0])
            else:
                scores.append(self.minValue(gameState.generateSuccessor(agentIndex, a), depth, (agentIndex+1))[0])
        minScore = min(scores)
        act = scores.index(minScore)

        return minScore, actions[act]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        # v = MAX-VALUE(state,-infinity,+infinity)
        value = None
        alpha = float("-inf")
        beta = float("inf")
        actions = gameState.getLegalActions(0)
        for a in actions:
            value = max(value, self.minValue(gameState.generateSuccessor(0, a), 1, 1, alpha, beta))
            # return the action in ACTIONS(state) with value v
            if alpha == float("-inf") or value > alpha:
                alpha = value
                action = a
        return action

    def maxValue(self, gameState, agentIndex, ply, alpha, beta):
        # if TERMINAL-TEST(state) then return UTILITY(state)
        if ply > self.depth:
            return self.evaluationFunction(gameState)
        # v = -infinity
        value = float("-inf")
        # for each a in ACTIONS(state) do
        actions = gameState.getLegalActions(agentIndex)
        for a in actions:
            # v = MAX(v, MIN-VALUE(RESULT(s,a), alpha, beta))
            value = max(value, self.minValue(gameState.generateSuccessor(agentIndex, a), (agentIndex + 1), ply, alpha, beta))
            # if v > beta then return v
            if value > beta:
                return value
            # alpha = MAX(alpha, v)
            alpha = max(alpha, value)
        # return v
        if value != float("-inf"):
            return value
        else:
            return self.evaluationFunction(gameState)

    def minValue(self, gameState, agentIndex, ply, alpha, beta):
        # if TERMINAL-TEST(state) then return UTILITY(state)
        if agentIndex == gameState.getNumAgents():
            return self.maxValue(gameState, 0, (ply + 1), alpha, beta)
        # v = infinity
        value = float("inf")
        # for each a in ACTIONS(state) do
        actions = gameState.getLegalActions(agentIndex)
        for a in actions:
            # v = MIN(v, MAX-VALUE(RESULT(s,a), alpha, beta))
            value = min(value, self.minValue(gameState.generateSuccessor(agentIndex, a), (agentIndex + 1), ply, alpha, beta))
            # if v <= alpha then return v
            if value < alpha:
                return value
            # beta = MIN(beta, v)
            beta = min(beta, value)
        # return v
        if value != float("inf"):
            return value
        else:
            return self.evaluationFunction(gameState)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        action = None
        # if terminal node
        if 0 == self.depth:
            return self.evaluationFunction(gameState)
        # if a max node
        if self.index == 0:
            actions = gameState.getLegalActions(self.index)
            value = float("inf")
            v = []
            for a in actions:
                v.append((self.expValue(gameState.generateSuccessor(self.index, a), 0, 1), a))
            value, action = max(v)

        return action

    def maxValue(self, gameState, depth, agentIndex):
        value = float("-inf")
        actions = gameState.getLegalActions(agentIndex)
        v = []
        if depth == self.depth or len(actions) == 0:
            return self.evaluationFunction(gameState)
        else:
            for a in actions:
                # values = [value(s') for s' in successors(s)]
                v.append(self.expValue(gameState.generateSuccessor(agentIndex, a), depth, (agentIndex + 1)))
            # return max(values)
            value = max(v)
        return value


    def expValue(self, gameState, depth, agentIndex):
        value = 0
        v = []
        actions = gameState.getLegalActions(agentIndex)
        if depth == self.depth or len(actions) == 0:
            return self.evaluationFunction(gameState)
        else:
            for a in actions:
                # values = [value(s') for s' in successors(s)]
                if agentIndex == gameState.getNumAgents() - 1:
                    v.append(self.maxValue(gameState.generateSuccessor(agentIndex, a), (depth + 1), 0))
                else:
                    v.append(self.expValue(gameState.generateSuccessor(agentIndex, a), depth, (agentIndex + 1)))
            # weight = [probability (s,s') for s' in successors(s)]
            # take the average
            weight = sum(v) / len(v)
        return weight

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
