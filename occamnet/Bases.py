from abc import ABC,abstractmethod
import torch
import sympy as sp
import numpy as np

#Nan represents unfixed units. Not wrong units.
def checkNan(input):
    return np.isnan(input[0])

#Inf represents wrong units that need to be propagated further
def checkInf(input):
    return np.isinf(input[0])

def matchUnits(unit1,unit2):
    if np.any(unit1 != unit2):
        if checkNan(unit1):
            return unit2
        if checkNan(unit2):
            return unit1
        return np.full(unit1.shape,np.inf)
    return unit1

def checkNonzero(unit):
    if np.any(unit):
        if not checkNan(unit):
            return np.full(unit.shape,np.inf)
        return np.zeros(unit.shape)
    return unit

class Base(ABC):
    @abstractmethod
    def getOutput(self, input):
        pass

    @abstractmethod
    def getSymbolicOutput(self, input):
        pass

    @abstractmethod
    def propagateUnits(self, input):
        pass

class BaseWithConstants(Base):
    def getConstants(self, constants):
        self.constants = constants
        #print(constants)
    
    def getSymbolicConstants(self,constants):
        self.symbolicConstants = constants

class Zero(Base):
    numInputs = 0

    def getLatex(self):
        return "0"

    def getOutput(self, input):
        return 0
    
    def getSymbolicOutput(self, input):
        return "0"
    
    def getSymbolicOutputConstant(self, input):
        return "0"

    def copy(self):
        return Zero()

    def propagateUnits(self, input):
        return np.full(input[0].shape, 0)

class One(Base):
    numInputs = 0

    def getLatex(self):
        return "1"

    def getOutput(self, input):
        return 1
    
    def getSymbolicOutput(self, input):
        return "1"
    
    def getSymbolicOutputConstant(self, input):
        return "1"

    def copy(self):
        return One()

    def propagateUnits(self, input):
        return np.full(input[0].shape, 0)

class ConstantN(Base):
    numInputs = 0

    def getLatex(self):
        return "N"

    def getOutput(self, input):
        return 20e6
    
    def getSymbolicOutput(self, input):
        return "N"
    
    def getSymbolicOutputConstant(self, input):
        return "N"

    def copy(self):
        return ConstantN()

    def propagateUnits(self, input):
        return np.full(input[0].shape, 0)


class Add(Base):
    numInputs = 2

    def getLatex(self):
        return "+"

    def getOutput(self, input):
        return input[:,0]+input[:,1]
    
    def getSymbolicOutput(self, input):
        return "("+str(input[0])+"+"+str(input[1])+")"
    
    def getSymbolicOutputConstant(self, input):
        return "("+str(input[0])+"+"+str(input[1])+")"

    def copy(self):
        return Add()

    def propagateUnits(self, input):
        return matchUnits(input[0],input[1])


class Add3(Base):
    numInputs = 3

    def getLatex(self):
        return "+"

    def getOutput(self, input):
        return input[:,0]+input[:,1]+input[:,2]
    
    def getSymbolicOutput(self, input):
        return "("+str(input[0])+"+"+str(input[1])+"+"+str(input[2])+")"
    
    def getSymbolicOutputConstant(self, input):
        return "("+str(input[0])+"+"+str(input[1])+"+"+str(input[2])+")"

    def copy(self):
        return Add3()

    def propagateUnits(self, input):
        input1 = matchUnits(input[0],input[1])
        return matchUnits(input[2],input1)


class Subtract(Base):
    numInputs = 2

    def getLatex(self):
        return "-"

    def getOutput(self, input):
        return input[:,0]-input[:,1]

    def getSymbolicOutput(self, input):
        return "("+str(input[0])+"-"+str(input[1])+")"

    def getSymbolicOutputConstant(self, input):
        return "("+str(input[0])+"-"+str(input[1])+")"

    def copy(self):
        return Subtract()

    def propagateUnits(self, input):
        return matchUnits(input[0],input[1])


class Multiply(Base):
    numInputs = 2

    def getLatex(self):
        return "×"

    def getOutput(self, input):
        return input[:,0]*input[:,1]

    def getSymbolicOutput(self, input):
        return "("+str(input[0])+"*"+str(input[1])+")"

    def getSymbolicOutputConstant(self, input):
        return "("+str(input[0])+"*"+str(input[1])+")"
    
    def copy(self):
        return Multiply()

    def propagateUnits(self, input):
        if checkInf(input[0]):
            return input[0]
        if checkInf(input[1]):
            return input[1]
        return input[0]+input[1]

class Divide(Base):
    numInputs = 2

    def getLatex(self):
        return "÷"

    def getOutput(self, input):
        return input[:,0]/input[:,1]

    def getSymbolicOutput(self, input):
        return "("+str(input[0])+"/"+str(input[1])+")"

    def getSymbolicOutputConstant(self, input):
        return "("+str(input[0])+"/"+str(input[1])+")"

    def copy(self):
        return Divide()
    
    def propagateUnits(self, input):
        if checkInf(input[0]):
            return input[0]
        if checkInf(input[1]):
            return input[1]
        return input[0]-input[1]

class Sin(Base):
    numInputs = 1

    def getLatex(self):
        return "cos"

    def getOutput(self, input):
        return torch.sin(input[:,0])

    def getSymbolicOutput(self, input):
        return "sin("+str(input[0])+")"

    def getSymbolicOutputConstant(self, input):
        return "sin("+str(input[0])+")"

    def copy(self):
        return Sin()

    def propagateUnits(self, input):
        return checkNonzero(input[0])

class Cos(Base):
    numInputs = 1

    def getLatex(self):
        return "cos"

    def getOutput(self, input):
        return torch.cos(input[:,0])

    def getSymbolicOutput(self, input):
        return "cos("+str(input[0])+")"

    def getSymbolicOutputConstant(self, input):
        return "cos("+str(input[0])+")"

    def copy(self):
        return Cos()

    def propagateUnits(self, input):
        return checkNonzero(input[0])

class MultiplyHundred(Base):
    numInputs = 1

    def getLatex(self):
        return "100"

    def getOutput(self, input):
        return 100*input[:,0]
    
    def getSymbolicOutput(self, input):
        return "(100*"+str(input[0])+")"
    
    def getSymbolicOutputConstant(self, input):
        return "(100*"+str(input[0])+")"

    def copy(self):
        return MultiplyHundred()

    def propagateUnits(self, input):
        return input[0]

class MultiplyThousand(Base):
    numInputs = 1

    def getLatex(self):
        return "1000"

    def getOutput(self, input):
        return 1000*input[:,0]
    
    def getSymbolicOutput(self, input):
        return "(1000*"+str(input[0])+")"
    
    def getSymbolicOutputConstant(self, input):
        return "(1000*"+str(input[0])+")"

    def copy(self):
        return MultiplyThousand()

    def propagateUnits(self, input):
        return input[0]

class Square(Base):
    numInputs = 1

    def getLatex(self):
        return "x^2"

    def getOutput(self, input):
        return input[:,0]*input[:,0]

    def getSymbolicOutput(self, input):
        return "("+str(input[0])+"^2)"

    def getSymbolicOutputConstant(self, input):
        return "("+str(input[0])+"^2)"

    def copy(self):
        return Square()

    def propagateUnits(self, input):
        return 2*input[0]

class Cube(Base):
    numInputs = 1

    def getLatex(self):
        return "x^3"

    def getOutput(self, input):
        return input[:,0]*input[:,0]*input[:,0]

    def getSymbolicOutput(self, input):
        return "("+str(input[0])+"^3)"

    def getSymbolicOutputConstant(self, input):
        return "("+str(input[0])+"^3)"

    def copy(self):
        return Cube()

    def propagateUnits(self, input):
        return 3*input[0]

    
class Log(Base):
    numInputs = 1

    def getLatex(self):
        return "\ln"

    def getOutput(self, input):
        return torch.log(input[:,0])

    def getSymbolicOutput(self, input):
        return "ln("+str(input[0])+")"

    def getSymbolicOutputConstant(self, input):
        return "ln("+str(input[0])+")"

    def copy(self):
        return Log()

    def propagateUnits(self, input):
        return checkNonzero(input[0])
    
class Exp(Base):
    numInputs = 1

    def getLatex(self):
        return "\exp"

    def getOutput(self, input):
        return torch.exp(input[:,0])

    def getSymbolicOutput(self, input):
        return "exp("+str(input[0])+")"

    def getSymbolicOutputConstant(self, input):
        return "exp("+str(input[0])+")"

    def copy(self):
        return Exp()

    def propagateUnits(self, input):
        return checkNonzero(input[0])

class AddConstant(BaseWithConstants):
    numInputs = 1
    numConstants = 1

    def getLatex(self):
        return "+c"

    def getOutput(self, input):
        return input[:,0]+self.constants[0]

    def getSymbolicOutput(self, input):
        if type(self.constants[0]) is not int:
            return "("+str(input[0])+"+"+str(round(self.constants[0].item(),6))+")"
        return "("+str(input[0])+"+"+str(round(self.constants[0],6))+")"

    def getSymbolicOutputConstant(self, input):
        return "("+str(input[0])+"+"+self.symbolicConstants[0]+")"

    def copy(self):
        return AddConstant()

    def propagateUnits(self, input):
        return input[0]

class MultiplyConstant(BaseWithConstants):
    numInputs = 1
    numConstants=1
    
    def getLatex(self):
        return "*c"

    def getOutput(self, input):
        return input[:,0]*self.constants[0]

    def getSymbolicOutput(self, input):
        if type(self.constants[0]) is not int:
            return "("+str(round(self.constants[0].item(),6))+"*"+str(input[0])+")"
        return "("+str(round(self.constants[0],6))+"*"+str(input[0])+")"

    def getSymbolicOutputConstant(self, input):
        return "("+self.symbolicConstants[0]+"*"+str(input[0])+")"
    
    def copy(self):
        return MultiplyConstant()
    
    def propagateUnits(self, input):
        return np.full(input[0].shape, np.nan)

class PowerConstant(BaseWithConstants):
    numInputs = 1
    numConstants=1
    
    def getLatex(self):
        return "x^c"

    def getOutput(self, input):
        output = torch.pow(input[:,0],self.constants[0])
        if torch.any(output!=output):
            return output.detach()
        return output

    def getSymbolicOutput(self, input):
        if type(self.constants[0]) is not int:
            return "("+str(input[0])+"^"+str(round(self.constants[0].item(),6))+")"
        return "("+str(input[0])+"^"+str(round(self.constants[0],6))+")"

    def getSymbolicOutputConstant(self, input):
        return "("+str(input[0])+"^"+self.symbolicConstants[0]+")"

    def copy(self):
        return PowerConstant()

    def propagateUnits(self, input):
        return checkNonzero(input[0])

class Quadratic(BaseWithConstants):
    numInputs = 1
    numConstants = 3
    
    def getLatex(self):
        return "ax^2+bx+c"

    def getOutput(self, input):
        output = self.constants[0]*(input[:,0]**2)+self.constants[1]*input[:,0]+self.constants[2]
        if torch.any(output!=output):
            return output.detach()
        return output

    def getSymbolicOutput(self, input):
        if type(self.constants[0]) is not int:
            return f"({round(self.constants[0].item(),6)}*{input[0]}^2+{round(self.constants[1].item(),6)}*{input[0]}+{round(self.constants[2].item(),6)})"
        return f"({round(self.constants[0],6)}*{input[0]}^2+{round(self.constants[1],6)}*{input[0]}+{round(self.constants[2],6)})"

    def getSymbolicOutputConstant(self, input):
        return f"({self.symbolicConstants[0]}*{input[0]}^2+{self.symbolicConstants[1]}*{input[0]}+{self.symbolicConstants[2]})"

    def copy(self):
        return Quadratic()

    def propagateUnits(self, input):
        return np.full(input[0].shape, np.nan)
    
class LogNorm(BaseWithConstants):
    numInputs = 1
    numConstants = 2
    
    def getLatex(self):
        return "\operatorname{LogNorm}(x; \mu, \sigma)"

    def getOutput(self, input):
        output = 1/(torch.FloatTensor(input[:,0])*self.constants[1]*np.sqrt(2*np.pi))*torch.exp(-(torch.log(input[:,0])-self.constants[0])**2/(2*self.constants[1]**2))

        if torch.any(output!=output):
            return output.detach()
        return output

    def getSymbolicOutput(self, input):
        if type(self.constants[0]) is not int:
            return f"LogNorm({input[0]}; {round(self.constants[0].item(),6)}, {round(self.constants[1].item(),6)})"
        return f"LogNorm({input[0]}; {round(self.constants[0],6)}, {round(self.constants[1],6)})"

    def getSymbolicOutputConstant(self, input):
        return f"LogNorm({input[0]}; {self.symbolicConstants[0]}, {self.symbolicConstants[1]})"

    def copy(self):
        return LogNorm()

    def propagateUnits(self, input):
        return np.full(input[0].shape, np.nan)

class LogNormExp(BaseWithConstants):
    numInputs = 1
    numConstants = 2
    
    def getLatex(self):
        return "\exp(\frac{(\ln(x)-\mu)^2}{2\sigma^2})"

    def getOutput(self, input):
        output = torch.exp(-(torch.log(input[:,0])-self.constants[0])**2/(2*self.constants[1]**2))

        if torch.any(output!=output):
            return output.detach()
        return output

    def getSymbolicOutput(self, input):
        if type(self.constants[0]) is not int:
            return f"exp((-(ln({input[0]})-{round(self.constants[0].item(),6)})^2)/(2*{round(self.constants[1].item(),6)}^2))"
        return f"exp((-(ln({input[0]})-{round(self.constants[0],6)})^2)/(2*{round(self.constants[1],6)}^2))"

    def getSymbolicOutputConstant(self, input):
        return f"exp((-(ln({input[0]})-{self.symbolicConstants[0]})^2)/(2*{self.symbolicConstants[1]}^2))"

    def copy(self):
        return LogNormExp()

    def propagateUnits(self, input):
        return np.full(input[0].shape, 0)

class NormExp(BaseWithConstants):
    numInputs = 1
    numConstants = 2
    
    def getLatex(self):
        return "\exp(\frac{(x-\mu)^2}{2\sigma^2})"

    def getOutput(self, input):
        output = torch.exp(-(input[:,0]-self.constants[0])**2/(2*self.constants[1]**2))

        if torch.any(output!=output):
            return output.detach()
        return output

    def getSymbolicOutput(self, input):
        if type(self.constants[0]) is not int:
            return f"exp((-({input[0]}-{round(self.constants[0].item(),6)})^2)/(2*{round(self.constants[1].item(),6)}^2))"
        return f"exp((-({input[0]}-{round(self.constants[0],6)})^2)/(2*{round(self.constants[1],6)}^2))"

    def getSymbolicOutputConstant(self, input):
        return f"exp((-({input[0]}-{self.symbolicConstants[0]})^2)/(2*{self.symbolicConstants[1]}^2))"

    def copy(self):
        return NormExp()

    def propagateUnits(self, input):
        return np.full(input[0].shape, 0)

class Beta(BaseWithConstants):
    numInputs = 1
    numConstants = 3
    
    def getLatex(self):
        return "\Beta_0(1+\Beta_1\cos(\frac{2\pi t}{T}-\phi))"

    def getOutput(self, input):
        output = self.constants[0]*(1 + self.constants[1]*torch.cos(2*np.pi*input[:,0]/52-self.constants[2]))

        if torch.any(output!=output):
            return output.detach()
        return output

    def getSymbolicOutput(self, input):
        if type(self.constants[0]) is not int:
            return f"{round(self.constants[0].item(),6)}(1 + {round(self.constants[1].item(),6)}cos(2\u03C0{input[0]}/T-{round(self.constants[2].item(),6)}))"
        return f"{round(self.constants[0],6)}(1 + {round(self.constants[1],6)}cos(2\u03C0{input[0]}/T-{round(self.constants[2],6)}))"

    def getSymbolicOutputConstant(self, input):
        return f"{self.symbolicConstants[0]}(1 + {self.symbolicConstants[1]}cos(2\u03C0{input[0]}/T-{self.symbolicConstants[2]}))"


    def copy(self):
        return Beta()

    def propagateUnits(self, input):
        return np.full(input[0].shape, np.nan)
