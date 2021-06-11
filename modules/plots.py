import numpy as np
import matplotlib.pyplot as plt

def plot_train(history, epochs):
    fig = plt.figure(figsize=(10, 10))
    plt.title('Training and validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(np.arange(0, epochs), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), history.history["val_loss"], label="val_loss")

    plt.title('Training and validation accuracy')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(np.arange(0, epochs), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), history.history["val_accuracy"], label="val_acc")
    return fig

def plot_augmented_batch(train_generator):
    # Print some examples of random data agumentation
    batchx, batchy = next(train_generator)

    for i in range(batchx.shape[0]):
        fig, ax = plt.subplots(1, 3)
        ax[0].set_title('image')
        ax[0].imshow(batchx[i][:,:,0], cmap='gray')
        ax[1].set_title('label')
        ax[1].imshow(batchy[i][:,:,0], cmap='gray')
        ax[2].set_title('image+label')
        ax[2].imshow(batchx[i][:,:,0], cmap='gray')
        ax[2].imshow(batchy[i][:,:,0], 'jet', interpolation='none', alpha=0.5)
    return fig

def plot_test_results(test_results):

    fig = plt.figure(figsize=(15, 15))

    fig.add_subplot(2,2,1)
    plt.title("Test Image")
    plt.imshow(test_results['test_image'][0,:,:,0], cmap='gray')

    fig.add_subplot(2,2,2)
    plt.title("Test Mask")
    plt.imshow(test_results['test_mask'], cmap='gray')


    fig.add_subplot(2,2,3)
    plt.title("Predicted Image")
    plt.imshow(test_results['pred_image'][0,:,:,0], cmap='gray')

    fig.add_subplot(2,2,4)
    plt.title("Predicted Mask")
    plt.imshow(test_results['pred_mask'], cmap='gray')

    return fig