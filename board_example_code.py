import numpy as np
import chess

'''
This is some of the code from when I was working on this project on my own.
Its not very clean and doesn't follow standard conventions because I have a large background in javascript/node
It should give a good starting point for encapsulating the chess library in a gym env
https://github.com/openai/gym
https://github.com/openai/gym/tree/master/gym/envs#how-to-create-new-environments-for-gym
http://python-chess.readthedocs.io/en/latest/index.html
'''


# base class for interacting with board
class baseBoard(chess.Board):

    def __init__(self):
        #chess.Board() is the base object in the chess library for interacting with the board.
        self.baseBoard = chess.Board()
        self.parsedBoard = self.parseBoard()
        #moveStack was meant to store moves for tree searching several moves ahead 
        self.moveStack = []
        #I was storing training data inside this object just because it was convinient
        self.trainingData = []
        self.legalMoveList = self.getLegalMoveList()

    '''
    parseBoard was written this way so that I could easily create new classes that parse the board differently without breaking anything
    All I would need to do is overwrite the parseBoard method with a new parsing of the board
    For the gym encapsulation there should just be a default board parsing with white pawns as 1, black pawns as -1, knights as +-2 etc
    And different parsings of the board should be handled by an abstraction layer
    '''
    def parseBoard(self):
        self.parsedBoard = self.baseBoard

    #It's very important that state observations include a list of legal moves
    def getLegalMoveList(self):
        a = list(enumerate(self.baseBoard.legal_moves))
        b = [x[1] for x in a]
        c = []
        i = 0
        for item in b:
            c.append(str(b[i]))
            i += 1
        return c

    #This just allows for moves to be represented as either a single uci string or a pair of squares. Again this should be an abstraction layer
    def parseMove(self, fromSquare, toSquare=None):
        frT = type(fromSquare)
        toT = type(toSquare)
        if(toSquare is None and frT == str):
            move = fromSquare
        elif(frT == str and toT == str):
            move = fromSquare + toSquare
        else:
            return "Invalid Move Representation"
        return move

    #The main thing to note here is that I append q to the uci string if a move isn't legal because otherwise pawns couldn't move past the 7th rank
    #I've not been able to come up with any way for a network to represent underpromotion
    def playMove(self, move):
        if(move in self.legalMoveList):
            self.baseBoard.push_uci(move)
        elif(move + 'q' in self.legalMoveList):
            self.baseBoard.push_uci(move + 'q')
        else:
            return 'illegal move'

        self.updateAttributes()
        if(self.baseBoard.is_game_over()):
            return self.baseBoard.result()

        return self.parsedBoard
    
    def addTrainingData(self, move):
        a = [self.parsedBoard, move]
        if(self.playMove(move) != 'illegal move'):
            self.trainingData.append(a)
            return self.parsedBoard
        else:
            return 'illegal move'

    #just resets the board to the starting position
    def reset(self):
        self.baseBoard.reset()
        self.updateAttributes()
        return self.parsedBoard

    #This is called after every move to update the board representation and legal move list, not very elegant and there's definitely a better way
    def updateAttributes(self):
        self.legalMoveList = self.getLegalMoveList()
        self.parseBoard()



#This functions the same as the baseBoard class but gives a different parsing of the board state
class parsedBoard(baseBoard):

    #Here all pieces are seperated into different slices of an array so one slice is white pawns, one is black bishops etc
    def parseBoard(self):
        b0 = np.zeros((12, 8, 8))
        i = 0
        #Very messy code
        while(i < 8):
            j = 0
            while(j < 8):
                k = str(self.baseBoard.piece_at(i * 8 + j))
                #The representation changes depending on who's turn it is so the same slice of the array always refers to that sides pieces
                if(self.baseBoard.turn):
                    if(k == "P"):
                        b0[0][i][j] = 1
                    elif(k == "B"):
                        b0[1][i][j] = 1
                    elif(k == "N"):
                        b0[2][i][j] = 1
                    elif(k == "R"):
                        b0[3][i][j] = 1
                    elif(k == "Q"):
                        b0[4][i][j] = 1
                    elif(k == "K"):
                        b0[5][i][j] = 1
                    elif(k == "p"):
                        b0[6][i][j] = 1
                    elif(k == "b"):
                        b0[7][i][j] = 1
                    elif(k == "n"):
                        b0[8][i][j] = 1
                    elif(k == "r"):
                        b0[9][i][j] = 1
                    elif(k == "q"):
                        b0[10][i][j] = 1
                    elif(k == "k"):
                        b0[11][i][j] = 1
                else:
                    if(k == "P"):
                        b0[6][i][j] = 1
                    elif(k == "B"):
                        b0[7][i][j] = 1
                    elif(k == "N"):
                        b0[8][i][j] = 1
                    elif(k == "R"):
                        b0[9][i][j] = 1
                    elif(k == "Q"):
                        b0[10][i][j] = 1
                    elif(k == "K"):
                        b0[11][i][j] = 1
                    elif(k == "p"):
                        b0[0][i][j] = 1
                    elif(k == "b"):
                        b0[1][i][j] = 1
                    elif(k == "n"):
                        b0[2][i][j] = 1
                    elif(k == "r"):
                        b0[3][i][j] = 1
                    elif(k == "q"):
                        b0[4][i][j] = 1
                    elif(k == "k"):
                        b0[5][i][j] = 1
                j = j + 1
            i = i + 1
        return b0


board = parsedBoard()
