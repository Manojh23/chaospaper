import math
import torch
import torch.nn.functional as F

def npo2(L):
    """ Next power of 2 above L """
    return 2 ** math.ceil(math.log2(L))

def pad_npo2(X):
    """
    Pads input (B, L, D, N) along 'L' to the next power of 2
    Returns shape (B, npo2(L), D, N)
    """
    len_npo2 = npo2(X.size(1))
    pad_tuple = (0, 0, 0, 0, 0, len_npo2 - X.size(1))
    return F.pad(X, pad_tuple, "constant", 0)

class PScan_H(torch.autograd.Function):
    @staticmethod
    def pscan(A, X, H):
        """
        A, X, H all have shape (B, D, L, N).
        This does the forward “upsweep” portion of the Blelloch scan,
        plus the hierarchical H projection at each step.
        """
        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        Aa = A
        Xa = X
        Ha = H
        for _ in range(num_steps - 2):
            T = Xa.size(2)  # "time" length
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)
            Ha = Ha.view(B, D, T // 2, 2, -1)

            # Combine and project
            Xa[:, :, :, 1].add_(Aa[:, :, :, 1] * Xa[:, :, :, 0])
            Xa[:, :, :, 1] = Ha[:, :, :, 1] * Xa[:, :, :, 1]
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]
            Ha = Ha[:, :, :, 1]

        # Handle final small L if needed
        if Xa.size(2) == 4:
            Xa[:, :, 1].add_(Aa[:, :, 1] * Xa[:, :, 0])
            Xa[:, :, 1] = Ha[:, :, 1] * Xa[:, :, 1]
            Aa[:, :, 1].mul_(Aa[:, :, 0])

            Xa[:, :, 3].add_(Aa[:, :, 3] * (Xa[:, :, 2] + Aa[:, :, 2] * Xa[:, :, 1]))
            Xa[:, :, 3] = Ha[:, :, 3] * Xa[:, :, 3]
        elif Xa.size(2) == 2:
            Xa[:, :, 1].add_(Aa[:, :, 1] * Xa[:, :, 0])
            Xa[:, :, 1] = Ha[:, :, 1] * Xa[:, :, 1]
        # else no action

    @staticmethod
    def pscan_rev(A, X, H):
        """
        A, X, H also have shape (B, D, L, N).
        This does the backward “downsweep,” but currently only updates
        X in place. (Full chain-rule for A,H is not implemented here.)
        """
        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        Aa = A
        Xa = X
        Ha = H
        for _ in range(num_steps - 2):
            T = Xa.size(2)
            Aa = Aa.view(B, D, T // 2, 2, -1)
            Xa = Xa.view(B, D, T // 2, 2, -1)
            Ha = Ha.view(B, D, T // 2, 2, -1)

            Xa[:, :, :, 0].add_(Aa[:, :, :, 0] * Xa[:, :, :, 1])
            Xa[:, :, :, 0] = Ha[:, :, :, 0] * Xa[:, :, :, 0]
            Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])

            Aa = Aa[:, :, :, 0]
            Xa = Xa[:, :, :, 0]
            Ha = Ha[:, :, :, 0]

        if Xa.size(2) == 4:
            Xa[:, :, 2].add_(Aa[:, :, 2] * Xa[:, :, 3])
            Xa[:, :, 2] = Ha[:, :, 2] * Xa[:, :, 2]
            Aa[:, :, 2].mul_(Aa[:, :, 3])

            Xa[:, :, 0].add_(
                Aa[:, :, 0] * (Xa[:, :, 1] + Aa[:, :, 1] * Xa[:, :, 2])
            )
            Xa[:, :, 0] = Ha[:, :, 0] * Xa[:, :, 0]
        elif Xa.size(2) == 2:
            Xa[:, :, 0].add_(Aa[:, :, 0] * Xa[:, :, 1])
            Xa[:, :, 0] = Ha[:, :, 0] * Xa[:, :, 0]
        # else no action

    @staticmethod
    def forward(ctx, A_in, X_in, H_in):
        """
        forward: expects A_in, X_in, H_in in shape (B, L, D, N).
        1) Possibly pad along L to nearest power of 2
        2) Transpose them to (B, D, L, N)
        3) Run pscan forward
        4) Return final shape (B, L, D, N)
        """
        L = X_in.size(1)

        # 1) Pad if needed
        if L == npo2(L):
            A = A_in.clone()
            X = X_in.clone()
            H = H_in.clone()
        else:
            A = pad_npo2(A_in)
            X = pad_npo2(X_in)
            H = pad_npo2(H_in)

        # 2) Transpose to (B, D, L, N)
        A = A.transpose(2, 1)
        X = X.transpose(2, 1)
        H = H.transpose(2, 1)

        # 3) Forward pass
        PScan_H.pscan(A, X, H)

        # 4) Save some for backward. We store:
        #    - A_in in (B, L, D, N)
        #    - X and H in (B, D, L, N)
        ctx.save_for_backward(A_in, X, H)

        # Return X in shape (B, L, D, N), slicing off any pad
        return X.transpose(2, 1)[:, :L]

    @staticmethod
    def backward(ctx, grad_output_in):
        """
        In backward, we must:
          1) retrieve saved A_in (B,L,D,N), X, H (B,D,L,N)
          2) pad if needed
          3) transpose grad_output & A_in to (B,D,L,N) so everything matches
          4) run pscan_rev
          5) build partial grads for (A_in, X_in, H_in), each in original shape
        """
        A_in, X, H = ctx.saved_tensors
        # A_in is (B, L, D, N)
        # X, H are (B, D, L, N)

        L = grad_output_in.size(1)

        # 1) Pad if needed
        if L != npo2(L):
            grad_output = pad_npo2(grad_output_in)     # (B, L_npo2, D, N)
            A_in_padded = pad_npo2(A_in)               # (B, L_npo2, D, N)
        else:
            grad_output = grad_output_in.clone()
            A_in_padded = A_in

        # 2) Transpose grad_output & A_in to (B,D,L,N)
        grad_output = grad_output.transpose(2, 1)   # (B, D, L_npo2, N)
        A_in_padded = A_in_padded.transpose(2, 1)   # (B, D, L_npo2, N)
        # X, H are already (B, D, L_npo2, N) from the forward save

        # 3) We slice off the first index along the L dimension from A_in
        #    (the same trick from the forward pass), then pad in the last dimension
        #    so pscan_rev matches shape expectations
        A = F.pad(A_in_padded[:, :, 1:], (0, 0, 0, 1))  # (B, D, L_npo2, N)

        # 4) Backward pass
        PScan_H.pscan_rev(A, grad_output, H)

        # 5) Build partial derivatives
        # Q for partial derivative wrt. X (or A_in) from the formula:
        #    Q[:, :, 1:] += X[:, :, :-1] * grad_output[:, :, 1:]
        Q = torch.zeros_like(X)  # (B, D, L_npo2, N)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])

        # We are returning (gradA_in, gradX_in, gradH_in).
        #
        # Original shapes are (B, L, D, N). So we do:
        gradA = Q.transpose(2, 1)[:, :L]          # => (B, L, D, N)
        gradX = grad_output.transpose(2, 1)[:, :L]
        gradH = H.transpose(2, 1)[:, :L]

        return gradA, gradX, gradH

pscan_H = PScan_H.apply
