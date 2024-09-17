def show_result(gen_A, gen_B, images):
     #validate by test images
    #print 5 real A -> generated B -> generated A
    #print 5 real B -> generated A -> generated B


    _, axes = plt.subplots(5, 3, figsize=(10, 10))
    axes = np.reshape(axes, (15, ))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    for i in range(5):
        real_A = images[0][i]
        fake_B = gen_B(real_A.to(device))
        cycle_A = gen_A(fake_B.to(device))
        real_A = real_A.numpy().transpose((1, 2, 0))
        real_A =  std * real_A + mean
        fake_B = fake_B.cpu().data.numpy().transpose((1, 2, 0))
        fake_B = std * fake_B + mean
        cycle_A = cycle_A.cpu().data.numpy().transpose((1, 2, 0))
        cycle_A = std * cycle_A + mean

        axes[3*i].imshow(real_A)
        axes[3*i].axis('off')
        axes[3*i+1].imshow(fake_B)
        axes[3*i+1].axis('off')
        axes[3*i+2].imshow(cycle_A)
        axes[3*i+2].axis('off')
    plt.show()

    _, axes = plt.subplots(5, 3, figsize=(10, 10))
    axes = np.reshape(axes, (15, ))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    for i in range(5):
        real_B = images[1][i]
        fake_A = gen_A(real_B.to(device))
        cycle_B = gen_B(fake_A.to(device))
        real_B = real_B.numpy().transpose((1, 2, 0))
        real_B =  std * real_B + mean
        fake_A = fake_A.cpu().data.numpy().transpose((1, 2, 0))
        fake_A = std * fake_A + mean
        cycle_B = cycle_B.cpu().data.numpy().transpose((1, 2, 0))
        cycle_B = std * cycle_B + mean

        axes[3*i].imshow(real_B)
        axes[3*i].axis('off')
        axes[3*i+1].imshow(fake_A)
        axes[3*i+1].axis('off')
        axes[3*i+2].imshow(cycle_B)
        axes[3*i+2].axis('off')
    plt.show()
