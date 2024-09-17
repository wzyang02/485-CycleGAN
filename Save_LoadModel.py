def saveModel(gen_A, gen_B, disc_A, disc_B, optGen, optDisc):
    #
    # SAVE training model including optimizer in case of taining is interrupted
    #  model can continue to train
    #
    if SAVE_MODEL is False:
        return
    print("Saving model ....")
    #save generator
    data = {"state_dict": gen_A.state_dict(), "optimizer": optGen.state_dict()}
    torch.save(data, "genA.pth.tar")
    data = {"state_dict": gen_B.state_dict(), "optimizer": None}
    torch.save(data, "genB.pth.tar")
    #save discrimotor
    data = {"state_dict": disc_A.state_dict(), "optimizer": optDisc.state_dict()}
    torch.save(data, "discA.pth.tar")
    data = {"state_dict": disc_B.state_dict(), "optimizer": None}
    torch.save(data, "discB.pth.tar")


def loadModel(gen_A, gen_B, disc_A, disc_B, optGen, optDisc):
    #
    # find out how many epochs already trained
    #
    with open("Lossfile","r") as outfile:
        prevData = json.load(outfile)
        epoch = len(prevData["G_LOSS"])

    print("start epoch {} loading model ....".format(epoch))
    #load generator
    data = torch.load("genA.pth.tar", map_location=device)
    gen_A.load_state_dict(data["state_dict"])
    optGen.load_state_dict(data["optimizer"])
    data = torch.load("genB.pth.tar", map_location=device)
    gen_B.load_state_dict(data["state_dict"])
    #optGen.load_state_dict(data["optimizer"])
    #load discriminator
    data = torch.load("discA.pth.tar", map_location=device)
    disc_A.load_state_dict(data["state_dict"])
    optDisc.load_state_dict(data["optimizer"])
    data = torch.load("discB.pth.tar", map_location=device)
    disc_B.load_state_dict(data["state_dict"])
    #optDisc.load_state_dict(data["optimizer"])

    for param_group in optGen.param_groups:
        param_group["lr"] = LEARNING_RATE
    for param_group in optDisc.param_groups:
        param_group["lr"] = LEARNING_RATE


    return epoch