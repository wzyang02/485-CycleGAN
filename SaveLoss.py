#plotting
with open("Lossfile","r") as outfile:
    prevData = json.load(outfile)
    saveLoss_G = prevData["G_LOSS"]
    saveLoss_D = prevData["D_LOSS"]
    saveLoss_C = prevData["C_LOSS"]
    saveLoss_I = prevData["I_LOSS"]

epochs = int(len(saveLoss_G))
x = [i+1 for i in range(epochs)]

plt.plot(x, saveLoss_G, label = 'Generator Losses')
plt.plot(x, saveLoss_D, label = 'Discriminator Losses')
plt.plot(x, saveLoss_C, label = 'Cycle Consistency Losses')
plt.plot(x, saveLoss_I, label = 'Identity Losses')
plt.xticks(range(0,epochs, 20))
plt.xlabel('epochs values')
plt.legend()
plt.show()
