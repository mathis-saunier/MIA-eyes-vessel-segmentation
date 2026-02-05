import matplotlib.pyplot as plt

def viz_train_and_val_losses(train_losses, val_losses):
    nb = max(len(train_losses), len(val_losses))
    plt.plot(range(1, nb + 1), train_losses,color='green')
    plt.plot(range(1, nb + 1), val_losses,color='pink')
    plt.show()