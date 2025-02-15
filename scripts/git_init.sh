
git config --global user.email "1016509070@qq.com"


git config --global user.name "NJUxlj"

eval "$(ssh-agent -s)"


ssh-add ../.ssh/dsw-key-1


ssh -T git@github.com

git remote set-url origin git@github.com:NJUxlj/blackprompt-bidirectional-autocot-prompt-tuning.git


# git remote set-url origin git@github.com:NJUxlj/Travel-Agent-based-on-LLM-and-SFT.git