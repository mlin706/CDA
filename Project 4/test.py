import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython import clear_output, display

def train_model(model, num_epochs, train_loader, device, criterion, optimizer):
    epoch_losses = []

    plt.ion()  # interactive mode for live updates
    fig, ax = plt.subplots(figsize=(6, 4))
    line, = ax.plot([], [], marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average training loss (MSE)")
    ax.set_title("Training loss convergence")
    ax.grid(True)

    for epoch in range(num_epochs):

        epoch_loss = 0.0
        n_batches = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            dynamic_ncols=True,
        )

        for X_batch, y_batch in progress_bar:
            X_batch = X_batch.to(device) # 1 mini-batch of input features, shape [batch_size, 4]
            y_batch = y_batch.to(device) # corresponding precipitation targets, shape [batch_size, 1]

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backpropagation and parameter update
            optimizer.zero_grad()   # clears accumulated gradients
            loss.backward()         # computes gradients via chain rule
            optimizer.step()        # updates parameters using Adam

            # Accumulate statistics
            epoch_loss += loss.item()
            n_batches += 1

            progress_bar.set_postfix(
                batch_loss=f"{loss.item():.4f}"
            )

        avg_loss = epoch_loss / n_batches if n_batches > 0 else float("nan")
        epoch_losses.append(avg_loss)

        # ---- Live plot update ----
        clear_output(wait=True)
        ax.clear()
        ax.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Average training loss (MSE)")
        ax.grid(True)
        display(fig)

        print(f"Epoch {epoch+1}, Avg Loss: {avg_loss:.4f}")