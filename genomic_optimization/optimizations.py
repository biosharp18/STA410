import torch
import numpy as np

def get_gradient_direction(cur_pos, target_base):
    base_dict = {0: np.array([1, 0, 0, 0]),
                 1: np.array([0, 1, 0, 0]),
                 2: np.array([0, 0, 1, 0]),
                 3: np.array([0, 0, 0, 1])}
    direction_vector = base_dict[target_base] - cur_pos
    norm = np.linalg.norm(direction_vector)
    return direction_vector / (norm if norm != 0 else 1) 

def gradient_ascent(X, grad):
    edits = np.ones(X.shape[2]) * -1
    return X + grad, edits

def directional_ascent(X, grad):
    edits = np.ones(X.shape[2]) * -1
    for i in range(X.shape[2]):
        current_pos = X[0,:,i]
        current_grad = grad[0,:,i]
        directional_gradients = np.ones(4) * -1e4 

        for target_base in [0,1,2,3]:
            direction = get_gradient_direction(current_pos, target_base)
            directional_gradients[target_base] = np.dot(current_grad, direction)

        max_grad = np.max(directional_gradients)
        max_direction = np.argmax(directional_gradients)
        if max_grad > 0:
            X[0,:,i] = X[0,:,i] + directional_gradients[max_direction] * get_gradient_direction(current_pos, max_direction)
    return X, edits

def simplex_ascent(X, grad):
    edits = np.ones(X.shape[2]) * -1
    normal_vec = np.array([1/2,1/2,1/2,1/2])
    #print(np.repeat(normal_vec[:,np.newaxis], 2114, 1).shape)
    edited_grad = grad - np.einsum("ijk,j->ik", grad, normal_vec)*np.repeat(normal_vec[:,np.newaxis], 2114, 1)
    edited_grad = edited_grad.astype(np.float32)
    return X + edited_grad, edits



def run_gradient_optimization(X, model, update_function, n_iterations=300, base_to_log=1005, softmax=False):
    """
    X: torch tensor of shape (1, 4, 2114)
    update_function
    """
    optimization_results = np.zeros(n_iterations)
    one_hot_results = np.zeros(n_iterations)
    # Store edits at each iteration
    optimization_edits = np.zeros((n_iterations, X.shape[2]))
    #Log trajectory of just one base
    trajectory = np.zeros((n_iterations, 4))
    

    X_grad = X.grad.detach().cpu().numpy()
    X_copy = X.cpu().detach().clone().numpy()

    for i in range(n_iterations):
        X_new, _ = update_function(X_copy, X_grad)
        X_new = torch.tensor(X_new, device='cuda', requires_grad=True)
        
        if softmax:
            X_new_sm = torch.nn.functional.softmax(X_new, dim=1)
            y=model(X_new_sm)
        else:
            y=model(X_new)
        optimization_results[i] = y.detach().cpu().numpy()
        y.backward()

        max_Xs = np.argmax(X_new.detach().cpu().numpy(), axis=1)
        optimization_edits[i:i+1, :] = max_Xs
        trajectory[i:i+1, :] = X_new[0, :, base_to_log].detach().cpu().numpy()
        
        one_hot_ = np.eye(4)[max_Xs].transpose(0, 2, 1).astype(np.float32)
        one_hot_X = torch.from_numpy(one_hot_).to('cuda')
        one_hot_X.requires_grad = True
        
        one_hot_y = model(one_hot_X)
        one_hot_results[i] = one_hot_y.detach().cpu().numpy()

        # Update tracking variables
        X_grad = X_new.grad.detach().cpu().numpy()
        X_copy = X_new.detach().cpu().numpy()
    return optimization_results, one_hot_results, optimization_edits, trajectory
        