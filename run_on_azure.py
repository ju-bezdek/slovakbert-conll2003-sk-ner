import argparse
from azureml.core import Workspace, ScriptRunConfig, Environment, Experiment, ComputeTarget
from azureml.core.runconfig import MpiConfiguration
from azureml.core import Workspace
from azureml.core import Environment

#Constants
WS_CONFIG_PATH='.azureml'
COMPUTE_TARGET="Standard-NC6-Promo"

def run_remote():
    print(f"Run remote train... OK? [Y/N]")
    if input().upper()!="Y":
        print("Exiting...")
    
    ws=Workspace.from_config(WS_CONFIG_PATH)

    env = Environment.from_pip_requirements('slovakbert_conll2003_sk_ner', './src/requirements.txt')
    
    #distr_config = MpiConfiguration(process_count_per_node=1, node_count=1)
    
    compute_target = ComputeTarget(workspace=ws, name=COMPUTE_TARGET)
    instance_status = compute_target.get_status().state
    if instance_status=="Stopped":
        compute_target.start(wait_for_completion=True, show_output=True)
    elif instance_status!="Running" and instance_status!="JobRunning":
        print(f"Unable to run experiment. Instance status: {instance_status}")
        return
    run_config = ScriptRunConfig(

        source_directory= './src',
        script='train.py',
        compute_target=compute_target,
        environment=env,
        #distributed_job_config=distr_config,
        arguments = [
        #'--learning_rate', 0.001,
        #'--momentum', 0.9,
        ] 
    )

    # submit the run configuration to start the job
    run = Experiment(ws, "experiment_name").submit(run_config)
    print("Follow progress here:")
    print(run.get_portal_url())
    run.wait_for_completion(show_output=True)
    files = run.get_file_names()
    for f in files:
        if "checkpoint" not in f:
            print(f"downloading: {f}")
            run.download_file(name=f, output_file_path=f)

    compute_target.stop(wait_for_completion=False, show_output=True)


def create_ws(subscribtion_id:str, workspace_name:str, resource_group:str = "ML", location:str = "westeurope"):
    print(f"Create workspace {workspace_name} at resource_group {resource_group}, location={location}... OK? [Y/N]")
    if input()!="Y":
        print("Exiting...")
    if workspace_name in Workspace.list(subscribtion_id):
        print(f"{workspace_name} already exists")
        ws = Workspace.get(workspace_name, subscription_id=subscribtion_id)
        ws.write_config(path=WS_CONFIG_PATH)
    else:
        ws = Workspace.create(name=workspace_name, # provide a name for your workspace
                            subscription_id=subscribtion_id, # provide your subscription ID
                            resource_group=resource_group, # provide a resource group name
                            #create_resource_group=True,
                            location=location) # e.g. 'westeurope' or 'eastus2' or 'westus2' or 'southeastasia'.

        # write out the workspace details to a configuration file: .azureml/config.json
        ws.write_config(path=WS_CONFIG_PATH)

def create_compute_intance(cpu_cluster_name:str):
    
    from azureml.core.compute import ComputeTarget, AmlCompute
    from azureml.core.compute_target import ComputeTargetException

    ws = Workspace.from_config() # automatically looks for a directory .azureml/

    # name for your cluster
    cpu_cluster_name = "Standard-NC6-Promo"

    try:
        # check if cluster already exists
        cpu_cluster = ComputeTarget(workspace=ws, name="Standard-NC6-Promo")
        print('Found existing cluster, use it.')
    except ComputeTargetException:
        # if not, create it
        compute_config = AmlCompute.provisioning_configuration(
            vm_size='Standard_NC6_Promo',
            max_nodes=4, 
            idle_seconds_before_scaledown=2400,)
        cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)
        cpu_cluster.wait_for_completion(show_output=True)

def get_ws():
    Workspace.from_config(WS_CONFIG_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Run remote train job on azure')
    subparsers = parser.add_subparsers(help="create_ws | run_remote", dest='command')
    create_ws_parser = subparsers.add_parser('create_ws', help='create azure ML workspace')
    
    create_ws_parser.add_argument('-sub_id', help="Subscription id", required=True)
    create_ws_parser.add_argument('-ws', required=True, help="workspace name")
    create_ws_parser.add_argument('-rg', required=True, help="resource group")
    create_ws_parser.add_argument('--location', default="westeurope")

    create_ws_parser = subparsers.add_parser('run_remote', help='run on azure')
  
    args = parser.parse_args()
    print(args.command)
    print(args)

    if args.command=="create_ws":
        create_ws(args.sub_id, args.ws, args.rg, args.location)
    elif args.command=="run_remote":
        run_remote()
    