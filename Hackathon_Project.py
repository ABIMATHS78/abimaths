#This code was written by Mojeed Damilola Abisiga
#Problem Definition:
#Two friends say, Erica and Bob participate in a friendly Hackathon that allows each to solve one question a day out of the specified number of questions offered.
#There will be one easy, one medium, and one hard, with points awarded based on difficulty. The participant can also choose to skip a question. The winner is the
#one with the highest score at the of the competition based on the following scale. In case of a tie, the person who solves the most hard problems wins. If it's
#the same, the one with more medium problems wins; otherwise, the one with more easy problems wins. If they both have the same score and the same number of problems
#at the same difficulty level, then it's a tie.
#               Scoring Table
#___________________________________________
#Diificulty            |             Points
#   Skip (S)           |                0
#   Easy (E)           |                1
#   Medium (M)         |                3
#   Hard(H)            |                5
#This program also calculate some basic statistics of the games played between Erica and Bob. It will also generate a scatter plot and histogram of the total points acquired by
#Erica and Bob in each day of the Hackathon.
import numpy as np
import matplotlib.pyplot as plt
def value(x):                               #This Function takes care of the conditions in the Scoring Table
    y = []
    for i in range(len(x)):
        if x[i] == "S":
            a = 0
        elif x[i] == "E":
            a = 1
        elif x[i] == "M":
            a = 3
        elif x[i] == "H":
            a = 5
        else:
            print("Invalid Argument(s)")
            break
        y += [a]
    return y

def winner(erica, bob):                 #This function compares Erica and Bob Hackathon points for each day and returns the winner of the that day's Hackathon
    value(erica)
    value(bob)
    totalA = sum(value(erica))
    totalB = sum(value(bob))
    if totalA > totalB:
        return "Erica"
    elif totalA < totalB:
        return "Bob"
    else:
        if erica.count("H") > bob.count("H"):
            return "Erica"
        elif erica.count("H") < bob.count("H"):
            return "Bob"
        else:
            if erica.count("M") > bob.count("M"):
                return "Erica"
            elif erica.count("M") < bob.count("M"):
                return "Bob"
            else:
                if erica.count("E") > bob.count("E"):
                    return "Erica"
                elif erica.count("E") < bob.count("E"):
                    return "Bob"
                else:
                    if erica.count("S") > bob.count("S"):
                        return "Erica"
                    elif erica.count("S") < bob.count("S"):
                        return "Bob"
                    else:
                        return "Tie"
n = int(input("Enter number of questions you guys want for the Hackathon: "))
listA = []
listB = []
for i in range(n):
    erica = str(input("Enter erica's combination for day %d here: " % (i+1)))
    bob = str(input("Enter bob'S combination for day %d here: " % (i+1)))
    print(winner(erica, bob))
    totalA = sum(value(erica))
    totalB = sum(value(bob))
    listA += [totalA]
    listB += [totalB]
maxA = max(listA)
maxB = max(listB)
indA = listA.index(maxA)
indB = listB.index(maxB)
A = listA[indA]
B = listB[indB]
plt.scatter(listA, listB)
plt.xlabel('Erica total points for each game')
plt.ylabel('Bob total points for each game')
plt.title('A Scatter Plot of the total points of Erica and Bob for each Day of the Hackathon')
plt.text(maxA, B, 'Erica Maximum Total')
plt.text(A, maxB, 'Bob Maximum Total')
plt.grid(True)
plt.show()
plt.clf()
plt.hist(listA)
plt.show()
plt.clf()
plt.hist(listB)
plt.show()
plt.clf()
arrayA = np.array(listA)
arrayB = np.array(listB)
othersA = arrayA[arrayA != arrayB]
othersB = arrayB[arrayA != arrayB]
print("The mean of the Total points obtained by Erica is: %.4f" % np.mean(arrayA))
print("The mean of the Total points obtained by Bob is: %.4f" % np.mean(arrayB))
print("The maximum point obtained by Erica is: ", np.max(arrayA))
print("The maximum point obtained by Bob is: ", np.max(arrayB))
print("The minimum point obtained by Erica is: ", np.min(arrayA))
print("The minimum point obtained by Bob is: ", np.min(arrayB))
print("The median of the points obtained by Erica is: ", np.median(arrayA))
print("The median of the points obtained by Bob is: ", np.median(arrayB))
print("The Standard Deviation of the points obtained by Erica is: %.4f" % np.std(arrayA))
print("The Standard Deviation of the points obtained by Erica is: %.4f" % np.std(arrayB))
corr = np.corrcoef((arrayA, arrayB))
print("The degree of association(Correlation) betweeen Erica and Bob Total points: " + str(corr))
print("The mean of the points obtained by Erica, excluding where they both had the same Total points is: %.4f" % np.mean(othersA))
print("The mean of the points obtained by Bob, excluding where they both had the same Total points is: %.4f" % np.mean(othersB))
a = (np.sum(arrayA ** 2) * np.sum(arrayB) - np.sum(arrayA) * np.sum(arrayA * arrayB))/(len(arrayA) * np.sum(arrayA ** 2) - (np.sum(arrayA) ** 2))       #Logistic regression model from my knowledge of statistics
b = (len(arrayA) * (np.sum(arrayA * arrayB)) - np.sum(arrayA) * np.sum(arrayB))/(len(arrayA) * np.sum(arrayA ** 2) - (np.sum(arrayA) ** 2))
print("The logistic regression model is Y = %.4f + %.4f X" % (a, b))
predict = str(input("Enter Erica's combination to predict what Bob's total point will be: "))
list_predict = value(predict)
bob_predict = sum(list_predict)
print("Bob's predicted total point is: %.0f" % (a + b * bob_predict))               #This predicts the total point Bob when Erica states its combination
