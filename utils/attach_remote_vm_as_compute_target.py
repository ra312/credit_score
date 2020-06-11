from azureml.core.compute import RemoteCompute, ComputeTarget

# Create the compute config 
compute_target_name = "attach-dsvm"

attach_config = RemoteCompute.attach_configuration(
												#resource_id='<resource_id>',
                                                #ssh_port=22,
                                                #username='<username>',
                                                #password="<password>")

# If you authenticate with SSH keys instead, use this code:
                                                  ssh_port=22,
                                                  username='ra312',
                                                  password=None,
                                                  private_key_file="/Users/ra312/.ssh/id_rsa",
#                                                 private_key_passphrase="<passphrase>")

# Attach the compute
compute = ComputeTarget.attach(ws, compute_target_name, attach_config)

compute.wait_for_completion(show_output=True)
