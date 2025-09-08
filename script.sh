#!/bin/bash
set -e

echo "Starting Docker environment setup..."

export DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC LANG=C.UTF-8 LC_ALL=C.UTF-8 CUDA_HOME=/usr/local/cuda

echo "Updating package lists..."
apt-get update

echo "Installing base development tools..."
apt-get install -y --no-install-recommends \
  build-essential git curl wget ca-certificates \
  python3 python3-pip python3-venv \
  cmake ninja-build pkg-config \
  gdb clang clangd \
  ripgrep fd-find unzip zip \
  software-properties-common

# (optional) vim if you still want it; we'll use Neovim for coc.nvim
apt-get install -y vim
rm -rf /var/lib/apt/lists/*

echo "Installing Node.js 18..."
curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
apt-get update && apt-get install -y --no-install-recommends nodejs
rm -rf /var/lib/apt/lists/*

echo "Installing Python packages..."
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 torch torchvision
python3 -m pip install --no-cache-dir numpy

echo "Creating non-root user..."
USER_NAME=${1:-dev}
USER_UID=${2:-1000}
USER_GID=${3:-1000}

groupadd -g "${USER_GID}" "${USER_NAME}" 2>/dev/null || true
useradd  -m -u "${USER_UID}" -g "${USER_GID}" -s /bin/bash "${USER_NAME}" 2>/dev/null || true

mkdir -p /workspace

echo "Installing Neovim..."
curl -fsSL -o /tmp/nvim.tar.gz \
  https://github.com/neovim/neovim/releases/download/nightly/nvim-linux-x86_64.tar.gz

tar -C /opt -xzf /tmp/nvim.tar.gz
ln -sf /opt/nvim-linux-x86_64/bin/nvim /usr/local/bin/nvim

nvim --version | head -2

echo "Setting up editor config for ${USER_NAME}..."
set -e
cd ~

# vim-plug for Neovim (note the nvim path)
curl -fLo ~/.local/share/nvim/site/autoload/plug.vim --create-dirs \
  https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim

mkdir -p ~/.config/nvim

# Neovim config mirroring your plugins (coc requires modern nvim)
cat > ~/.config/nvim/init.vim <<'VIMRC'
set number
set tabstop=4 shiftwidth=4 expandtab
set termguicolors
syntax on
filetype plugin indent on

augroup cuda_ft
  autocmd!
  autocmd BufRead,BufNewFile *.cu,*.cuh set filetype=cpp
augroup END

call plug#begin(stdpath('data') . '/plugged')
  Plug 'neoclide/coc.nvim', {'branch': 'release'}
  Plug 'tpope/vim-fugitive'
  Plug 'preservim/nerdtree'
  Plug 'vim-airline/vim-airline'
  Plug 'vim-airline/vim-airline-themes'
  Plug 'octol/vim-cpp-enhanced-highlight'
  Plug 'dense-analysis/ale'
  Plug 'morhetz/gruvbox'
call plug#end()

" Gruvbox configuration
set background=dark
let g:gruvbox_contrast_dark = 'hard'
let g:gruvbox_improved_strings = 1
let g:gruvbox_improved_warnings = 1
set background=dark
set notermguicolors
set t_Co=256
let g:gruvbox_termcolors=256
colorscheme gruvbox

" Airline theme to match Gruvbox
let g:airline_theme = 'gruvbox'
let g:airline_powerline_fonts = 1

let g:ale_fix_on_save = 1
let g:ale_fixers = {'cpp': ['clang-format'], 'cuda': ['clang-format'], 'python': ['black']}

nnoremap <leader>n :NERDTreeToggle<CR>
inoremap <silent><expr> <Tab> pumvisible() ? "\<C-n>" : "\<Tab>"
nmap <leader>rn <Plug>(coc-rename)
nmap gd <Plug>(coc-definition)
nmap gr <Plug>(coc-references)
VIMRC

# Install plugins headlessly
nvim +PlugInstall +qall || true

# coc settings + helper
mkdir -p ~/.config/coc
cat > ~/.config/coc/settings.json <<'COCSET'
{
  "clangd.path": "/usr/bin/clangd",
  "python.defaultInterpreterPath": "/usr/bin/python3"
}
COCSET
cat > ~/setup_coc.sh <<'COCINST'
#!/usr/bin/env bash
nvim +'CocInstall -sync coc-clangd coc-pyright' +qall || true
echo "coc setup complete"
COCINST
chmod +x ~/setup_coc.sh

# Triton test
cat > ~/triton_hello.py <<'PY'
import triton, triton.language as tl, torch
@triton.jit
def add_kernel(X,Y,Z,N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0); offs = pid*BLOCK + tl.arange(0, BLOCK); mask = offs < N
    tl.store(Z + offs, tl.load(X + offs, mask=mask) + tl.load(Y + offs, mask=mask), mask=mask)
def main():
    n=1<<20
    x,y=torch.randn(n,device="cuda"),torch.randn(n,device="cuda")
    z=torch.empty_like(x)
    grid=lambda META: ((n+META["BLOCK"]-1)//META["BLOCK"],)
    add_kernel[grid](x,y,z,N=n,BLOCK=1024)
    print("ok", torch.allclose(z, x+y, atol=1e-4))
if __name__=="__main__": main()
PY

echo "Setup complete!"
echo "As ${USER_NAME}:"
echo "  ~/setup_coc.sh          # installs coc extensions"
echo "  nvim                    # open editor"
echo "  python3 ~/triton_hello.py  # GPU required"
