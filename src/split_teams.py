import os
import subprocess
import re
import tempfile
from pathlib import Path

from src.utils.general import repo_root

if __name__ == '__main__':
    teams_txt = repo_root / "teams.txt"

    assert teams_txt.exists(), "To split out teams, the 'teams.txt' file must be present in the repository root."

    with open(teams_txt, 'r') as inp:
        teams = inp.read()

    # regex split out the teams (item 1 is name, then 2 is team, then repeat)
    teams_lst = re.split("=+\n",teams)[1:]

    team_data = {}
    for i in range(len(teams_lst)//2):
        idx = 2*i
        team_name = teams_lst[idx].split(":")[-1].strip()
        team_str = teams_lst[idx+1]
        team_data[team_name] = team_str

    print(f"Found {len(team_data.keys())} teams")

    teams_out_dir = repo_root / 'packed_teams'

    temp_dir = tempfile.mkdtemp()
    teams_out_dir.mkdir(exist_ok=True, parents=True)
    cwd = os.getcwd()

    showdown_root = repo_root / 'pokemon-showdown'

    os.chdir(showdown_root)

    for name, team in team_data.items():
        out_file = teams_out_dir / f"{name}.txt"

        temp_output_path = Path(temp_dir) / f"{name}.txt"
        with open(temp_output_path, 'w') as outp:
            outp.write(team)

        read_file = subprocess.Popen(["type", f"{str(temp_output_path)}"], stdout=subprocess.PIPE, shell=True)
        output = subprocess.check_output(("node", "pokemon-showdown", "pack-team"), stdin=read_file.stdout)
        output = output.decode('utf-8')
        read_file.wait()
        with open(out_file, 'w') as outp:
            outp.write(output)

        print(f"Finished packing {name}")

    os.chdir(cwd)