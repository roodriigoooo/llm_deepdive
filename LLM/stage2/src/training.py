import time
from tqdm.auto import tqdm
import torch
from stage1.src.gpt2small import generate_text


def text_to_token_ids(text, tokenizer, allowed_special):
    """
    tensor.unsqueeze(dim) inserts a new axis (of size 1) at index dim.
    x = torch.tensor([10, 20, 30])       shape: [3]
    x0 = x.unsqueeze(0)                 shape: [1,3]
    x1 = x.unsqueeze(1)                 shape: [3,1]
    """
    allowed_special = allowed_special or ('<|endoftext|>')
    token_list = tokenizer.encode(text, allowed_special=set(allowed_special))
    ids = torch.tensor(token_list).unsqueeze(0)
    #unsqueeze turns a 1D sequence of token IDs into a 2D batch of size 1.
    #almost all pytorch nn.Modules (embeddings, transformers, etc.)
    # expect inputs of shape (batch_size, seq_len, ...)
    # even f we only have one example, we need to present it as a batch of size 1.
    return ids

def token_ids_to_text(token_ids, tokenizer):
    """
    tensor.squeeze(dim: optional) removes the axis at index dim if its size is 1.
    y = torch.zeros(1, 5, 1)         shape: [1,5,1]
    y0 = y.squeeze(0)               shape: [5,1]
    y1 = y.squeeze(2)               shape: [1,5]
    y2 = y.squeeze()               shape: [5] (all dims 1 are removed)
    """
    if not isinstance(token_ids, torch.Tensor):
        raise TypeError("token_ids must be a torch.Tensor")
    flat = token_ids.squeeze(0)
    #squeeze(0) just undoes the batch dimension we previously added,
    # giving back the raw token sequence.
    return tokenizer.decode(flat.tolist())

def calc_loss_cross_entropy(logits, target):
    """
    logits.shape == (batch_size, sequence_length, vocab_size)
    targets.shape == (batch_size, sequence_length)
    """
    flat_logits = logits.flatten(0, 1) #(batch_size*seq_len, vocab)
    flat_targets = target.flatten(0) #(batch_size*seq_len)
    return torch.nn.functional.cross_entropy(flat_logits, flat_targets)


def eval_loss(data_loader, model, device, max_batches=None):
    """
    run model.eval() and return average cross-entropy loss over up to max_batches(optional)
    """
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            if max_batches is not None and i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total += calc_loss_cross_entropy(logits, y).item()
            count += 1
    return float('nan') if count == 0 else total / count

def eval_losses(model, loaders, device, max_batches):
    model.eval()
    losses = {}
    with torch.no_grad():
        for split, loader in loaders.items():
            losses[split] = eval_loss(loader, model, device, max_batches)
    model.train()
    return losses

def generate_sample(model, tokenizer, device, prompt):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    input_ids = text_to_token_ids(prompt, tokenizer, allowed_special=None).to(device)
    with torch.no_grad():
        output_ids = generate_text(model=model, idx=input_ids, max_new_tokens=50, context_size=context_size)
    text = token_ids_to_text(output_ids, tokenizer).replace("\n", " ")
    print(text)
    model.train()

def train_model(model, train_loader, val_loader,
                optimizer, device, num_epochs,
                eval_freq, eval_iter, sample_prompt, tokenizer):
    history = {
        "steps": [],
        "train_loss": [],
        "val_loss": []
    }
    model.to(device)
    global_step = 0
    tokens_seen = 0


    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_start = time.time()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}')
        for x,y in pbar:
            global_step += 1
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = calc_loss_cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            tokens_seen += x.numel()
            pbar.set_postfix(loss=loss.item())

            #periodic evaluation
            if global_step % eval_freq == 0:
                #train_l = eval_loss(train_loader, model, device, max_batches=eval_iter)
                #val_l = eval_loss(val_loader, model, device, max_batches=eval_iter)
                losses = eval_losses(
                    model,
                    {'train': train_loader, 'val': val_loader},
                    device,
                    eval_iter
                )
                history['steps'].append(global_step)
                history['train_loss'].append(losses['train'])
                history['val_loss'].append(losses['val'])
                print(f'[Step {global_step}], Train CE={losses["train"]}, Val CE={losses["val"]}, Tokens seen={tokens_seen}')

                if sample_prompt and tokenizer:
                    print("--> Sample generation:")
                    generate_sample(model, tokenizer, device, sample_prompt)

        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch} done in {epoch_time} seconds')
    return history

def train_model_with_lr_warmup(model, train_loader, val_loader,
                optimizer, device, num_epochs,
                eval_freq, eval_iter, sample_prompt, tokenizer,
                peak_lr=0.01, init_lr=0.0001):
    history = {
        "steps": [],
        "train_loss": [],
        "val_loss": []
    }
    model.to(device)
    global_step = -1
    tokens_seen = 0

    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * 0.2)
    lr_increment = (peak_lr - init_lr) / warmup_steps

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_start = time.time()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}')
        for x,y in pbar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            global_step += 1
            #lr_warmup edit
            if global_step < warmup_steps:
                lr = init_lr + global_step * lr_increment
            else:
                lr = peak_lr
            #apply
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            logits = model(x)
            loss = calc_loss_cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            tokens_seen += x.numel()
            pbar.set_postfix(loss=loss.item())

            #periodic evaluation
            if global_step % eval_freq == 0:
                #train_l = eval_loss(train_loader, model, device, max_batches=eval_iter)
                #val_l = eval_loss(val_loader, model, device, max_batches=eval_iter)
                losses = eval_losses(
                    model,
                    {'train': train_loader, 'val': val_loader},
                    device,
                    eval_iter
                )
                history['steps'].append(global_step)
                history['train_loss'].append(losses['train'])
                history['val_loss'].append(losses['val'])
                print(f'[Step {global_step}], Train CE={losses["train"]}, Val CE={losses["val"]}, Tokens seen={tokens_seen}')

                if sample_prompt and tokenizer:
                    print("--> Sample generation:")
                    generate_sample(model, tokenizer, device, sample_prompt)

        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch} done in {epoch_time} seconds')
    return history

