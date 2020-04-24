import matplotlib.pyplot as plt
import numpy as np
import random
import copy

class GANN:
    def __init__(self, architecture):
        self.architecture = architecture

        self.bestWeight = []
        self.ultimateBestScore = 0

        self.bestScores = []
        self.meanScores = []

        self.weights = []
        self.rewards = []
        self.size = 0

    # random weights for initial population
    def generatePopulation(self, size):
        self.size = size
        for _ in range(size):
            weight = []

            for j in range(len(self.architecture)-1):
                weight.append(np.random.rand(self.architecture[j+1], self.architecture[j]))

            self.weights.append(weight)

        self.rewards = np.zeros((size))
        #print(self.weights)

    # forward propagate inputs through kth weight in population (-1 to use the ultimate best weights, used once evolution is complete)
    def forwardPropagate(self, inputs, k=-1):
        if k == -1:
            weights = self.bestWeight
        else:
            weights = self.weights[k]
        if len(inputs) != self.architecture[0]:
            return False

        z = inputs
        for i in range(len(weights)):
            z = self.tanh(np.dot(weights[i], z))

        return z

    # modify rewards of kth weight in population based on environment
    def updateRewards(self, reward, k):
        self.rewards[k] += reward
        #print(self.rewards)

    # pick best from current population and perform mutations on them to create next generation
    def evolve(self, percentage, mutationFactor, save=False):
        if save:
            pass
        else:
            best = []

            self.meanScores.append(np.mean(self.rewards))

            for i in range(int(len(self.weights)*percentage)):
                bestIndex = np.argmax(self.rewards)

                best.append(self.weights[bestIndex])

                if i == 0:
                    self.bestScores.append(self.rewards[bestIndex])

                if i == 0 and self.rewards[bestIndex]>=self.ultimateBestScore:
                    self.bestWeight = self.weights[bestIndex]
                    self.ultimateBestScore = self.rewards[bestIndex]

                self.rewards = np.delete(self.rewards, bestIndex)
                del self.weights[bestIndex]

            self.weights = self.mutate(best, mutationFactor)
        
    # subroutine to mutate a population
    def mutate(self, best, mutationFactor):
        newWeights = []

        count = 0

        while(len(newWeights) < self.size):
            newWeights.append(copy.deepcopy(best[count]))

            if count >= len(best)-1:
                count = 0
            else:
                count += 1
        
        for i in range(len(newWeights) - len(best)):
            for _ in range(mutationFactor):
                randIndex1 = random.randint(0, len(newWeights[i])-1)
                randIndex2 = random.randint(0, len(newWeights[i][randIndex1])-1)
                randIndex3 = random.randint(0, len(newWeights[i][randIndex1][0])-1)

                newWeights[i][randIndex1][randIndex2][randIndex3] = random.random()

        return newWeights

    # reset rewards of population after one generation
    def reset(self):
        self.rewards = np.zeros((self.size))

    # various action functions for NN
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x) 

    # plot progree of best score and mean score for each generation
    def evolution(self):
        plt.xlabel('generations') 
        plt.ylabel('score')

        plt.plot(self.bestScores, label = "best score in population")
        plt.plot(self.meanScores, label = "mean score of population")

        plt.legend() 
        plt.show()
