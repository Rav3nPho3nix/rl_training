if [[ $# -ne 1 ]]; then
    echo "Il faut exactement un parametre : le numero de version"
    exit 1
fi

if ! [[ $1 =~ ^[0-9]+$ ]]; then
    echo "Le numero de version n'est pas un nombre"
    exit 1
fi

nom_dossier="custom_lite3_rear_balance_v$1"
echo "Creation du dossier '$nom_dossier'"
mkdir -p /home/crestic/stage_Giuliano/rl_training/logs/rsl_rl/${nom_dossier}
echo "OK"

echo "Copie du checkpoint pre-entraine dans /home/crestic/stage_Giuliano/rl_training/logs/rsl_rl/${nom_dossier}/pretrained"
# cp -dr /home/crestic/stage_Giuliano/rl_training/logs/rsl_rl/custom_lite3_long_jump/pretrained /home/crestic/stage_Giuliano/rl_training/logs/rsl_rl/${nom_dossier}/pretrained
cp -dr /home/crestic/stage_Giuliano/rl_training/logs/rsl_rl/deeprobotics_lite3_flat/pretrained/ /home/crestic/stage_Giuliano/rl_training/logs/rsl_rl/${nom_dossier}/pretrained/
echo "OK"
