
def train(gen_A, gen_B, disc_A, disc_B, optGen, optDisc, sGen, sDisc):
    A_reals = 0
    A_fakes = 0
    G_epoch_loss = 0
    D_epoch_loss = 0
    Cycle_epoch_loss = 0
    Identity_epoch_loss = 0
    loop = tqdm(trainLoader, leave=True)

    for idx, (real_A, real_B) in enumerate(loop):
        A = real_A.to(device)
        B = real_B.to(device)

        # Train Discriminators A and B
        with torch.cuda.amp.autocast():
            fake_A = gen_A(B)
            D_A_real = disc_A(A)
            D_A_fake = disc_A(fake_A.detach())
            A_reals += D_A_real.mean().item()
            A_fakes += D_A_fake.mean().item()
            D_A_real_loss = MSE(D_A_real, torch.ones_like(D_A_real))
            D_A_fake_loss = MSE(D_A_fake, torch.zeros_like(D_A_fake))
            D_A_loss = D_A_real_loss + D_A_fake_loss

            fake_B = gen_B(A)
            D_B_real = disc_B(B)
            D_B_fake = disc_B(fake_B.detach())
            D_B_real_loss = MSE(D_B_real, torch.ones_like(D_B_real))
            D_B_fake_loss = MSE(D_B_fake, torch.zeros_like(D_B_fake))
            D_B_loss = D_B_real_loss + D_B_fake_loss

            # put losses togethor
            D_loss = (D_A_loss + D_B_loss) / 2
            D_epoch_loss += D_loss

        optDisc.zero_grad()
        sDisc.scale(D_loss).backward()
        sDisc.step(optDisc)
        sDisc.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_A_fake = disc_A(fake_A)
            D_B_fake = disc_B(fake_B)
            loss_G_A = MSE(D_A_fake, torch.ones_like(D_A_fake))
            loss_G_B = MSE(D_B_fake, torch.ones_like(D_B_fake))
            genLoss = loss_G_A + loss_G_B

            # cycle loss
            cycle_A = gen_A(fake_B)
            cycle_B = gen_B(fake_A)
            cycleLoss = L1(A, cycle_A) + L1(B, cycle_B)
            Cycle_epoch_loss += cycleLoss

            # identity loss
            identity_A = gen_A(A)
            identity_B = gen_B(B)
            idLoss = L1(A, identity_A) + L1(B, identity_B)
            Identity_epoch_loss += idLoss
            # add all togethor
            G_loss = (genLoss + cycleLoss * CYCLE_WEIGHT + idLoss * IDENTITY_WEIGHT )

            G_epoch_loss += G_loss

        optGen.zero_grad()
        sGen.scale(G_loss).backward()
        sGen.step(optGen)
        sGen.update()

        if idx % 200 == 0:
            save_image(real_A * 0.5 + 0.5, f"saved_images/real_A_{idx}.png")
            save_image(fake_B * 0.5 + 0.5, f"saved_images/fake_B_{idx}.png")
            save_image(cycle_A * 0.5 + 0.5, f"saved_images/cycle_A_{idx}.png")

        loop.set_postfix(A_real=A_reals / (idx + 1), A_fake=A_fakes / (idx + 1))
    # return mean losses for one epoch
    return G_epoch_loss/(idx+1), D_epoch_loss/(idx+1),  Cycle_epoch_loss/(idx+1),  Identity_epoch_loss/(idx+1)

