import torch
import sys
from pong_env import PongEnv

INPUTS = 7
OUTPUTS = 3
ADD_CONNECTION_CHANCE = 0.1
ADD_NODE_CHANCE = 0.03
ITERATIONS = 1000
# SEED = 123123
SEED = 123
MAX_FRAMES = 2500
LOAD_BEST = True
BEST_FILE = "20_379_test.dat"
RENDER = True
LOAD = True
RENDER_BEST_UPDATES = True
PRINT_ACTIONS = True

# TODO replace this class with your model
class MyModelClass(torch.nn.Module):
    
    def __init__(self):
        pass
    
    def forward(self, x):
        pass

class Node:
    id = 0
    def __init__(self, layer, input=False):
        self.id = id
        self.reset()
        self.inputNode = input
        self.input = 0.0
        self.layer = layer
        Node.id += 1
    def forward(self, x):
        return torch.nn.Sigmoid()(x).item()
    def reset(self):
        self.input = 0.0
    def output(self):
        if self.inputNode:
            return self.forward(self.input)
        return self.input

class Connection:
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2
        # Randomize the weight on this connection
        self.weight = torch.rand([1]).item()
        self.layer = self.node1.layer
        if torch.randint(2, [1]).item() == 0:
            self.weight = -self.weight
    def forward(self):
        self.node2.input += self.weight * self.node1.output()
    def mutate(self, value=0):
        if value != 0:
            self.weight += value
            return
        self.weight = torch.rand([1]).item()
        if torch.randint(2, [1]).item() == 0:
            self.weight = -self.weight

# We will be using NEAT!
class PongPlayer(object):

    rewardHistory = []
    bestLayers = []
    bestConnections = []
    newBest = True
    def __init__(self, save_path, load=False):
        # self.build_model()
        # self.build_optimizer()
        self.populateInitial()
        self.reward = 0
        self.save_path = save_path
        if load:
            self.load()

    def populateInitial(self):
        self.connections = []
        self.layers = []
        # Create the default input and output layers
        self.layers.append([Node(0) for i in range(INPUTS)])
        self.layers.append([Node(1) for i in range(OUTPUTS)])
        # Create a random connection
        self.addConnection()

    def build_model(self):
        # TODO: define your model here
        # I would suggest creating another class that subclasses
        # torch.nn.Module. Then you can just instantiate it here.
        # your not required to do this but if you don't you should probably
        # adjust the load and save functions to work with the way you did it.
        self.model = MyModelClass()

    def build_optimizer(self):
        # TODO: define your optimizer here
        self.optimizer = None

    def addConnection(self, node1=None, node1I=0, node2=None, node2I=1):
        if not node1:
            node1I = torch.randint(low=0, high=node1I + 1, size=[1]).item()
            node1 = self.layers[node1I][torch.randint(len(self.layers[node1I]), [1]).item()]
        if not node2:
            node2I = torch.randint(low=node2I, high=len(self.layers), size=[1]).item()
            node2 = self.layers[node2I][torch.randint(len(self.layers[node2I]), [1]).item()]
        self.connections.append(Connection(node1, node2))

    def addNode(self):
        # need to add the node to be in the middle, can't be input or output
        ind = torch.randint(low=1, high=len(self.layers), size=[1]).item()
        if ind == len(self.layers) - 1:
            self.layers.insert(ind, [])
        
        node = Node(ind)
        # Need to add a connection between this node and another
        # And a node must go into this node
        self.layers[ind].append(node)
        # Add a connection going to this node
        self.addConnection(node2=node, node1I=ind)
        # Add a connection going out of this node
        self.addConnection(node1=node, node2I=ind + 1)

    def mutateConnection(self):
        # Choose a random connection and mutate it
        ind = torch.randint(len(self.connections), [1]).item()
        # self.connections[ind].mutate(torch.rand([1]).item() - 0.5)
        self.connections[ind].mutate()

    def get_action(self, state, reward):
        # TODO: this method should return the output of your model
        # First we forward through all connections (after setting the input nodes to contain the input state)
        for i in range(INPUTS):
            self.layers[0][i].input = state[i]
        for i in range(len(self.layers)):
            for c in self.connections:
                # SLOW!
                if c.layer != i:
                    continue
                c.forward()
        # Now get the maximum action of the outputs
        outs = [node.output() for node in self.layers[len(self.layers) - 1]]
        outs = torch.tensor(outs)
        action = torch.functional.argmax(outs).item()
        # print(action)
        self.reward = reward
        self.resetNodes()
        return action

    def resetNodes(self):
        for i in range(len(self.layers)):
            for node in self.layers[i]:
                node.reset()

    def resetPlayer(self):
        self.rewardHistory.append(self.reward)
        self.reward = 0

    def reset(self, i):
        # TODO: this method will be called whenever a game finishes
        # so if you create a model that has state you should reset it here
        # NOTE: this is optional and only if you need it for your model

        # Want to reset the player to have a new set of 'genes'
        print("Resetting player! Reward: " + str(self.reward) + " with overall max: " + str(max(self.rewardHistory) if len(self.rewardHistory) > 0 else "0"))
        if self.reward > max(self.rewardHistory) if len(self.rewardHistory) > 0 else 0:
            # New maximum reward! Don't change anything in this player.
            print("Setting best player!")
            self.setBest()
            if not LOAD_BEST:
                print("Saving best player to: " + str(i) + "_" + self.save_path)
                self.save(path=str(i) + "_" + self.save_path)
            self.resetPlayer()
            return
        self.resetPlayer()

        # Randomly choose to add a new connection
        if torch.rand([1]).item() < ADD_CONNECTION_CHANCE:
            print("Adding a connection!")
            self.addConnection()
        elif torch.rand([1]).item() < ADD_NODE_CHANCE:
            print("Adding a node!")
            self.addNode()
        else:
            print("Mutating a random connection!")
            self.mutateConnection()
        
        # print("Updated Nodes/Connections: " + str(self.layers) + "\nConnections: " + str(self.connections))

    def setBest(self):
        self.bestConnections = self.connections.copy()
        self.bestLayers = self.layers.copy()
        self.newBest = True

    def load(self):
        state = torch.load(self.save_path)
        if LOAD_BEST:
            state = torch.load(BEST_FILE)
            # print(state)
        if not state['layers'] or len(state['layers']) == 0 or len(state['layers'][0]) != INPUTS or len(state['layers'][len(state['layers'])-1]) != OUTPUTS:
            # print(state['layers'][0])
            print("COULD NOT LOAD FILE: " + self.save_path + " OR " + BEST_FILE)
            return
        self.rewardHistory = state['reward_history']
        self.layers = state['layers']
        self.connections = state['connections']
        # self.model.load_state_dict(state['model_state_dict'])
        # self.optimizer.load_state_dict(state['optimizer_state_dict'])

    def save(self, path=""):
        state = {
            'reward_history': self.rewardHistory,
            'layers': self.layers,
            'connections': self.connections
        }
        if path == "":
            torch.save(state, self.save_path)
        else:
            torch.save(state, path)

    
def play_game(player, render=True, load=False):
    global ITERATIONS
    # call this function to run your model on the environment
    # and see how it does
    if load:
        player.load()
        # Run it only once!
        if LOAD_BEST:
            ITERATIONS = 1
    for i in range(ITERATIONS):
        print("Starting iteration: " + str(i))
        print("Setting seed to: " + str(SEED))
        env = PongEnv()
        env.seed(SEED)
        state = env.reset()
        action = player.get_action(state, 0)
        done = False
        total_reward = 0
        frames = 0
        while not done and frames < MAX_FRAMES:
            next_state, reward, done, _ = env.step(action)
            if render:
                env.render()
            total_reward += reward
            if LOAD_BEST and PRINT_ACTIONS:
                s = "DOWN"
                if action == 1:
                    s = "STAY"
                elif action == 2:
                    s = "UP"
                print(s, flush=True)
            action = player.get_action(next_state, total_reward)
            frames += 1
        if frames == MAX_FRAMES:
            # Then the player gets a reward of 0 for this, because they got into a tie.
            print("Tie game! Game timed out!")
            # player.reward = 0
        env.close()
        if player.newBest and RENDER_BEST_UPDATES:
            print("Showing the new best player...")
            env = PongEnv()
            env.seed(SEED)
            state = env.reset()
            action = player.get_action(state, 0)
            done = False
            total_reward = 0
            frames = 0
            while not done and frames < MAX_FRAMES:
                next_state, reward, done, _ = env.step(action)
                env.render()
                total_reward += reward
                action = player.get_action(next_state, total_reward)
                frames += 1
            if frames == MAX_FRAMES:
                # Then the player gets a reward of 0 for this, because they got into a tie.
                print("Tie game! Game timed out!")
                # player.reward = 0
            env.close()
            player.newBest = False
        player.reset(i)
    print("Maximum after " + str(ITERATIONS) + " was reward: " + str(max(player.rewardHistory)))
    # if not load or not LOAD_BEST:
    #     print("Saving player to: " + player.save_path)
    #     player.save()
    
if __name__ == '__main__':
    play_game(PongPlayer("379_test.dat"), render=RENDER, load=LOAD)