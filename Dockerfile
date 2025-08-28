
# CUDA toolkit, nvcc, headers included
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    CUDA_HOME=/usr/local/cuda

# Base dev tools + Vim stack + Node (for coc.nvim)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl wget ca-certificates \
    python3 python3-pip python3-venv \
    cmake ninja-build pkg-config \
    gdb clang clangd \
    vim tmux ripgrep fd-find unzip zip \
    software-properties-common \
 && rm -rf /var/lib/apt/lists/*

# Node 18 (stable enough for coc.nvim)
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
 && apt-get update && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/*

# Python deps (Triton + friends)
RUN python3 -m pip install --upgrade pip setuptools wheel \
 && python3 -m pip install "triton" numpy

# Optional: PyTorch (uncomment if you want it here; big download)
# RUN python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision

# Create non-root user
ARG USER=dev
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} ${USER} && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USER}
USER ${USER}
WORKDIR /workspace

# Vim config + plugins (coc.nvim + clangd + pyright + basics)
RUN mkdir -p ~/.vim/autoload ~/.vim/plugged ~/.config/coc/extensions \
 && curl -fLo ~/.vim/autoload/plug.vim --create-dirs \
      https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim

# Basic .vimrc tuned for C++/CUDA & Python
RUN bash -lc 'cat > ~/.vimrc << "EOF"
set nocompatible
set number relativenumber
set tabstop=4 shiftwidth=4 expandtab
set termguicolors
syntax on
filetype plugin indent on

" Treat .cu/.cuh as C++
augroup cuda_ft
  autocmd!
  autocmd BufRead,BufNewFile *.cu,*.cuh set filetype=cpp
augroup END

call plug#begin("~/.vim/plugged")
  Plug "neoclide/coc.nvim", {"branch": "release"}
  Plug "tpope/vim-fugitive"
  Plug "preservim/nerdtree"
  Plug "vim-airline/vim-airline"
  Plug "octol/vim-cpp-enhanced-highlight"
  Plug "dense-analysis/ale" " static checks/format runners
call plug#end()

" ALE: use clang-format if available
let g:ale_fix_on_save = 1
let g:ale_fixers = {"cpp": ["clang-format"], "cuda": ["clang-format"], "python": ["black"]}
nnoremap <leader>n :NERDTreeToggle<CR>

" coc basics
inoremap <silent><expr> <Tab> pumvisible() ? "\<C-n>" : "\<Tab>"
nmap <leader>rn <Plug>(coc-rename)
nmap gd <Plug>(coc-definition)
nmap gr <Plug>(coc-references)
EOF'

# Install Vim plugins headlessly
RUN vim +PlugInstall +qall || true

# Install coc extensions
RUN node -e "process.exit(0)" \
 && vim +'CocInstall -sync coc-clangd coc-pyright' +qall || true

# Minimal Triton sanity file
RUN bash -lc 'cat > ~/triton_hello.py << "PY"
import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(X, Y, Z, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    y = tl.load(Y + offs, mask=mask)
    tl.store(Z + offs, x + y, mask=mask)

def main():
    n = 1 << 20
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    z = torch.empty_like(x)
    grid = lambda META: ((n + META["BLOCK"] - 1) // META["BLOCK"],)
    add_kernel[grid](x, y, z, N=n, BLOCK=1024)
    print("ok", torch.allclose(z, x + y, atol=1e-4))

if __name__ == "__main__":
    main()
PY'

CMD ["/bin/bash"]
