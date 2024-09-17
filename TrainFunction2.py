#write loss to file
def saveLoss(histLoss_G, histLoss_D, histLoss_C,histLoss_I):
    saveLoss_G = [y.item() for y in histLoss_G]
    saveLoss_D = [y.item() for y in histLoss_D]
    saveLoss_C = [y.item() for y in histLoss_C]
    saveLoss_I = [y.item() for y in histLoss_I]


    if LOAD_MODEL:
        with open("Lossfile","r") as outfile:
            prevData = json.load(outfile)
            saveLoss_G = prevData["G_LOSS"] + saveLoss_G
            saveLoss_D = prevData["D_LOSS"] + saveLoss_D
            saveLoss_C = prevData["C_LOSS"] + saveLoss_C
            saveLoss_I = prevData["I_LOSS"] + saveLoss_I

    lossData = {"G_LOSS": saveLoss_G, "D_LOSS": saveLoss_D, "C_LOSS": saveLoss_C, "I_LOSS": saveLoss_I}
    with open("Lossfile","w") as outfile:
        json.dump(lossData, outfile, indent=4)


NUM_EPOCHS = 200
LOAD_MODEL = True
SAVE_MODEL = True

#instantiate and itinialize models
discA = discriminator().to(device)
discA.weight_init(mean=0.0, std=0.02)
discB = discriminator().to(device)
discB.weight_init(mean=0.0, std=0.02)
genA = generator().to(device)
genA.weight_init(mean=0.0, std=0.02)
genB = generator().to(device)
genB.weight_init(mean=0.0, std=0.02)

histLoss_G = []
histLoss_D = []
histLoss_C = []
histLoss_I = []

#optimizers
optDisc = optim.Adam(
        list(discA.parameters()) + list(discB.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

optGen = optim.Adam(
        list(genA.parameters()) + list(genB.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

# schedulers
# during second 100 epochs, decay learning rate from 2e-4 to 2e-6
scheduler_G = lr_scheduler.LinearLR(optGen, start_factor=1.0, end_factor = 0.01, total_iters =100)
scheduler_D = lr_scheduler.LinearLR(optDisc, start_factor=1.0, end_factor = 0.01, total_iters =100)


startEpoch = 0
if LOAD_MODEL:
    startEpoch = loadModel(genA, genB, discA, discB, optGen, optDisc)

sGen = torch.cuda.amp.GradScaler()
sDisc = torch.cuda.amp.GradScaler()
start_time = time.time()
for epoch in range(startEpoch, NUM_EPOCHS):
    epoch_start_time = time.time()

    D_loss, G_loss,  Cycle_loss,  Identity_loss = train(genA, genB, discA, discB, optGen, optDisc, sGen, sDisc)

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('[%d/%d] - using time: %.2f seconds' % ((epoch + 1), NUM_EPOCHS, per_epoch_ptime))
    print('loss of discriminator D: %.3f, generator G: %.3f , cycle consistency: %.3f, identity consistency: %.3f'
          % (D_loss, G_loss, Cycle_loss, Identity_loss))

    histLoss_G.append(G_loss)
    histLoss_D.append(D_loss)
    histLoss_C.append(Cycle_loss)
    histLoss_I.append(Identity_loss)
    #save model and test model
    if SAVE_MODEL:
        saveModel(genA, genB, discA, discB, optGen, optDisc)
        saveLoss(histLoss_G, histLoss_D, histLoss_C, histLoss_I)
        if LOAD_MODEL:
            histLoss_G = []
            histLoss_D = []
            histLoss_C = []
            histLoss_I = []

    if epoch %50 == 0:
        with torch.no_grad():
            show_result(genA, genB,  images)

    if epoch > 100:
        before_lr = optDisc.param_groups[0]["lr"]
        scheduler_G.step()
        scheduler_D.step()
        after_lr = optDisc.param_groups[0]["lr"]
        print("Epoch %d: adam lr %.6f -> %.6f" % (epoch+1, before_lr, after_lr))
training_time = start_time -  time.time()

print('total training time for %d epoches : %.2f seconds' % (NUM_EPOCHS, training_time))
