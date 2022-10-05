import torch
import math

class CrossEntropyLoss:
    def __init__(self, std, topNumber, activationWeight=0, constantWeight=0, anomWeight = 0.2, badUnitWeight = 1):
        self.setStd(std)
        self.topNumber = topNumber
        self.weighting = torch.tensor([1.0/(n) for n in range(topNumber, 0, -1)])
        self.anomWeight = anomWeight
        self.activationWeight = activationWeight
        self.constantWeight = constantWeight
        self.badUnitWeight = badUnitWeight

    def setStd(self, std):
        self.std = std
        self.scaleFactor = 1/math.sqrt(2*math.pi*(std**2))
    
    def getError(self, y, predictions, numActivations, numConstants, indices=[], ensemble=False, numBadProbs = 0):
        gaussian = torch.distributions.Normal(y, self.std)

        predictions = predictions.permute(2,0,1)

        anom = (predictions != predictions)
        predictions[anom] = 0

        exp_err = torch.exp(gaussian.log_prob(predictions))
        N = y.shape[0]
        error = torch.zeros((predictions.shape[0], 1))
        if ensemble:
            for ind_pair in indices:
                start, end = ind_pair
                inds = torch.arange(start, end)
                curr_errs = torch.index_select(exp_err, 1, inds)
                curr_sum = torch.sum(curr_errs, 1)
                weight = (end-start)/N
                error += weight*curr_sum
        else:
            error = torch.sum(torch.exp(gaussian.log_prob(predictions)), 1)

        anom = torch.any(anom, 1)

        error -= self.scaleFactor * (self.activationWeight*numActivations.unsqueeze(1) + self.constantWeight*numConstants.unsqueeze(1))

        error[anom] = -self.anomWeight*self.scaleFactor

        if numBadProbs != 0:
            badProbsError = torch.full((numBadProbs,error.shape[1]),-self.badUnitWeight*self.scaleFactor)
            return torch.cat((error,badProbsError),axis=0)
        return error

    def getLossMultipleSamples(self, probs, y, predictions, numActivations, numConstants, indices=[], ensemble=False, badProbs = None):
        error = self.getError(y, predictions, numActivations, numConstants, indices=indices, ensemble=ensemble, numBadProbs=(0 if badProbs == None else badProbs.shape[0]))
        topLoc = torch.argsort(error, dim=0)[-self.topNumber:,:]
        error = torch.gather(error, 0, topLoc)
        if badProbs != None:
            probs = torch.cat((probs,badProbs),axis=0)
        probs = torch.gather(probs, 0, topLoc)
        result =  (torch.sum(torch.sum(-torch.log(probs)*error, axis=1)*self.weighting[0:probs.shape[0]]),torch.mean(error))
        return result

        
