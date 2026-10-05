#!/usr/bin/env python3

"""
Run : python remote_file_management.py --help

Key commands [CMD] :
 copy_data: Copy init script and FFCWS data
 download_results: Download json files to local machine
"""
import os
import shlex

import click
from az_helper import az_cli
from fabric import Connection


@click.group()
def cli1():
    pass


DEFAULT_USER = "millionmonkeys"


@cli1.command(context_settings={"show_default": True})
@click.argument("start_worker_id", type=int)
@click.argument("end_worker_id", type=int)
@click.option("--user", default=DEFAULT_USER)
@click.option("--rg", default="CSML-Credits-Budget-RG")
@click.option("--vm_prefix", default="mm-worker-")
def get_worker_ips(start_worker_id, end_worker_id, user, rg, vm_prefix):
    return _get_worker_ips(start_worker_id, end_worker_id, user, rg, vm_prefix)


def _get_worker_ips(start_worker_id, end_worker_id, user, rg, vm_prefix):
    hosts = []
    ansible_hosts = ["[compute]"]
    config_hosts = []
    ips = []
    for i in range(start_worker_id, end_worker_id + 1):
        # get ip address of this vm
        # az vm show -d -g  CSML-Credits-Budget-RG --name mm-worker-5 --query publicIps -o tsv
        vm_name = f"{vm_prefix}{i}"
        az_cmd = f"vm show -d -g {rg} --name {vm_name} --query publicIps -o tsv"
        res = az_cli(az_cmd)
        print(f"vm_name is {vm_name} and ip address is {res}")

        hosts.append(
            f"""
        [[server]]
            name = "{vm_name}"
            host = "{res}"
            user = "{user}"
        """
        )
        config_hosts.append(
            f"""
         Host {vm_name}
            HostName {res}
            User {user}
        """
        )
        ansible_hosts.append(f"""{vm_name} ansible_host={res}""")
        ips.append((vm_name, res))
    print("Writing ip addresses to servers.toml file")
    with open("servers.toml", "w") as w:
        w.write("\n".join(hosts))
    print("Writing ip addresses to hosts.ansi file")
    with open("hosts.ansi", "w") as w:
        w.write("\n".join(ansible_hosts))
    print("Writing ip addresses to id_config file")
    with open("id_config", "w") as w:
        w.write("\n".join(config_hosts))
    cssh_str = " ".join(f"{user}@{x[1]}" for x in ips)
    print(f"Run: --- clusterssh {cssh_str} --- to login all VMs simultaneously ")
    return hosts, ips


def put_file(cxn, local_path, remote_dir, fname):
    remote_path = os.path.join(remote_dir, fname)
    try:
        if remote_dir != "":
            cxn.run(f"mkdir -p {remote_dir} && touch {remote_path}")
        cxn.put(local_path, remote_path)
    except Exception as e:
        print(e)


@cli1.command(context_settings={"show_default": True})
@click.argument("start_worker_id", type=int)
@click.argument("end_worker_id", type=int)
@click.option("--user", default=DEFAULT_USER)
@click.option("--rg", default="CSML-Credits-Budget-RG")
@click.option("--vm_prefix", default="mm-worker-")
def copy_init(start_worker_id, end_worker_id, user, rg, vm_prefix):
    _copy_init(start_worker_id, end_worker_id, user, rg, vm_prefix)


def _copy_init(start_worker_id, end_worker_id, user, rg, vm_prefix):
    hosts, vms = [
        x for x in _get_worker_ips(start_worker_id, end_worker_id, user, rg, vm_prefix)
    ]
    print("\n".join(hosts))
    for name, ip in vms:
        # create fabric connection to this server
        connection = Connection(host=f"{user}@{ip}")
        print("Copying init to")
        print(user, name, ip)
        connection.put(os.path.expanduser("./init.sh"), None)


class FilesDefaultToFFCWS(click.Argument):
    def __init__(self, *args, **kwargs):
        kwargs["nargs"] = -1
        super().__init__(*args, **kwargs)

    def full_process_value(self, ctx, value):
        return super().process_value(
            ctx,
            value
            or (
                "FFMetadata_v09.csv",
                "FF_allwaves_2020v2.dta",
                "FF_wave6_2020v2.dta",
                "FFC_idnum.csv",
            ),
        )


@cli1.command(context_settings={"show_default": True})
@click.argument("src", cls=FilesDefaultToFFCWS)
@click.argument("start_worker_id", type=int)
@click.argument("end_worker_id", type=int)
@click.option("--user", default=DEFAULT_USER)
@click.option("--rg", default="CSML-Credits-Budget-RG")
@click.option("--vm_prefix", default="mm-worker-")
def copy_data(src, start_worker_id, end_worker_id, user, rg, vm_prefix):
    _copy_data(src, start_worker_id, end_worker_id, user, rg, vm_prefix)


def _copy_data(src, start_worker_id, end_worker_id, user, rg, vm_prefix):
    hosts, vms = [
        x for x in _get_worker_ips(start_worker_id, end_worker_id, user, rg, vm_prefix)
    ]
    dest_dir = "million-monkeys/data/dataset/fragile-families/"
    for name, ip in vms:
        # create fabric connection to this server
        connection = Connection(host=f"{user}@{ip}")
        print("Copying data to")
        print(user, name, ip)
        for src_fp in src:
            connection.put(
                os.path.realpath(
                    os.path.join("../data/dataset/fragile-families", src_fp)
                ),
                dest_dir,
            )


@cli1.command(context_settings={"show_default": True})
@click.argument("dirname")
@click.argument("localdir")
@click.argument("start_worker_id", type=int)
@click.argument("end_worker_id", type=int)
@click.option("--user", default=DEFAULT_USER)
@click.option("--rg", default="CSML-Credits-Budget-RG")
@click.option("--vm_prefix", default="mm-worker-")
def download_run_results(
    dirname, localdir, start_worker_id, end_worker_id, user, rg, vm_prefix
):
    _download_run_results(
        dirname, localdir, start_worker_id, end_worker_id, user, rg, vm_prefix
    )


def _download_run_results(
    dirname, localdir, start_worker_id, end_worker_id, user, rg, vm_prefix
):
    hosts, vms = [
        x for x in _get_worker_ips(start_worker_id, end_worker_id, user, rg, vm_prefix)
    ]
    print("\n".join(hosts))
    for name, ip in vms:
        connection = Connection(host=f"{user}@{ip}")
        local_path = os.path.join(localdir, name)
        os.makedirs(local_path)
        remote_src = f"{user}@{ip}:{dirname.rstrip('/')}/"
        local_dst = local_path
        local_cmd = (
            "rsync -Ppthvrz "
            f"{shlex.quote(remote_src)} "
            f"{shlex.quote(local_dst)}"
        )
        connection.local(local_cmd)


cli = click.CommandCollection(sources=[cli1])

if __name__ == "__main__":
    cli()
