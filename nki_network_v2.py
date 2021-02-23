import torch
import numpy as np
from scipy.stats import truncnorm
from bindsnet.network import Network
from bindsnet.learning import PostPre, MSTDPET
from bindsnet.learning.reward import MovingAvgRPE
from bindsnet.network.monitors import Monitor
from bindsnet.network.topology import Connection
from bindsnet.network.nodes import Input, LIFNodes


# symmetric parametric truncated normal unity zero-centered dist (LeCun style)
def truncated_normal(mean=1., std=0.05, max_dev=0.1, size=(1, 1), device='cuda'):
    return torch.tensor(mean + std * truncnorm.rvs(a=-max_dev, b=max_dev, size=size), dtype=torch.float32,
                        device=device)


def Return_nki_net(v1_neurons, m_neurons, dt, device, runtime):
    # Build network.

    # default update rule :
    upd_rl = MSTDPET
    # upd_rl = PostPre

    v1_side = int(np.sqrt(v1_neurons))
    v1_inh_side = v1_side // 2

    network = Network(dt=dt, batch_size=1, learning=True, reward_fn=MovingAvgRPE)

    thalamus = Input(shape=(1, v1_neurons), traces=True)
    thalamus_inh = LIFNodes(n=v1_neurons // 4, traces=True)

    v1_directions = Input(shape=(1, v1_neurons * 8), traces=True)
    v1_exc = LIFNodes(n=v1_neurons, traces=True)
    v1_inh = LIFNodes(n=v1_neurons // 4, traces=True)

    motor = LIFNodes(n=m_neurons * 3, traces=True)
    motor_inh = LIFNodes(n=int(m_neurons * 0.75), traces=True)

    layers = {
        'Thalamus_Input': thalamus,
        'Thalamus_Inh': thalamus_inh,

        'V1_Directions_Input': v1_directions,
        'V1_Exc': v1_exc,
        'V1_Inh': v1_inh,

        'Motor': motor,
        'Motor_Inh': motor_inh
    }

    # connections norms
    thalamus_recur = 1e-10
    thalamus_inh_recur = 1e-10
    thalamus_to_thalamus_inh = 1e-10
    thalamus_inh_to_thalamus = -1e-10
    thalamus_to_v1_exc = 1.5

    v1_exc_recur = 1e-10
    v1_inh_recur = 1e-10
    v1_exc_to_v1_inh = 1e-10
    v1_inh_to_v1_exc = -1e-10
    v1_exc_to_motor = 1e-10

    v1_dir_recur = 1e-10
    v1_dir_to_motor = 1e-10

    motor_to_v1_exc = 1e-10
    motor_to_v1_dir = 1e-10
    motor_to_motor_inh = 1e-10
    motor_inh_to_motor = -1e-10
    motor_inh_recur = 1e-10

    l_rate = 0.0

    # Add all layers and connections to the network.
    for layer in layers:
        network.add_layer(layers[layer], name=layer)

    def recur_conn(name, strength):
        w = torch.ones(layers[name].n, layers[name].n, device=device) - torch.eye(layers[name].n, dtype=torch.float32,
                                                                                  device=device)
        network.add_connection(
            Connection(
                source=layers[name],
                target=layers[name],
                w=w * truncated_normal(size=w.shape), wmin=0., wmax=2. * strength,
                update_rule=upd_rl,
                nu=[l_rate * 1e-2, l_rate * 1e-2],
                norm=strength * layers[name].n,
            ),
            source=name, target=name
        )

    def all_to_all(name1, name2, strength):
        w = torch.ones(layers[name1].n, layers[name2].n, dtype=torch.float32, device=device)
        network.add_connection(
            Connection(
                source=layers[name1],
                target=layers[name2],
                w=w * truncated_normal(size=w.shape), wmin=0., wmax=2. * strength,
                norm=strength * layers[name1].n,
                update_rule=upd_rl,
                nu=[l_rate * 1e-2, l_rate * 1e-2],
            ),
            source=name1, target=name2
        )

    # RECURRENT CONN
    recur_conn('Thalamus_Input', thalamus_recur)
    recur_conn('Thalamus_Inh', thalamus_inh_recur)
    recur_conn('V1_Exc', v1_exc_recur)
    recur_conn('V1_Inh', v1_inh_recur)
    recur_conn('Motor_Inh', motor_inh_recur)

    # THALAMUS INPUT -> THALAMUS INH
    Source = 'Thalamus_Input'
    Target = 'Thalamus_Inh'
    w = torch.zeros(layers[Source].n, layers[Target].n, dtype=torch.float32, device=device)
    for i in range(v1_inh_side):
        src_i = min(i * 2, v1_side - 3)
        for j in range(v1_inh_side):
            src_j = min(j * 2, v1_side - 3)
            for di in range(3):
                for dj in range(3):
                    w[(src_j + dj) * v1_side + src_i + di, j * v1_inh_side + i] = 1
    network.add_connection(
        Connection(
            source=layers[Source],
            target=layers[Target],
            w=w * truncated_normal(size=w.shape),
            wmin=0.,
            wmax=2. * thalamus_to_thalamus_inh,
            norm=thalamus_to_thalamus_inh * layers[Source].n,
            update_rule=upd_rl,
            nu=[l_rate * 1e-2, l_rate * 1e-2],
        ),
        source=Source, target=Target
    )

    # THALAMUS INH -> THALAMUS INPUT
    Source = 'Thalamus_Inh'
    Target = 'Thalamus_Input'
    w = torch.zeros(layers[Source].n, layers[Target].n, dtype=torch.float32, device=device)
    for i in range(v1_inh_side):
        dst_i = min(i * 2, v1_side - 5)
        for j in range(v1_inh_side):
            dst_j = min(j * 2, v1_side - 5)
            for di in range(5):
                for dj in range(5):
                    w[j * v1_inh_side + i, (dst_j + dj) * v1_side + dst_i + di] = 1
    network.add_connection(
        Connection(
            source=layers[Source],
            target=layers[Target],
            w=w * truncated_normal(size=w.shape),
            wmin=0.,
            wmax=2. * thalamus_inh_to_thalamus,
            norm=thalamus_inh_to_thalamus * layers[Source].n,
            update_rule=upd_rl,
            nu=[l_rate * 1e-2, l_rate * 1e-2],
        ),
        source=Source, target=Target
    )

    # THALAMUS INPUT -> V1 EXC
    Source = 'Thalamus_Input'
    Target = 'V1_Exc'
    w = torch.zeros(layers[Source].n, layers[Target].n, dtype=torch.float32, device=device)
    for i in range(v1_side):
        src_i = min(i, v1_side - 3)
        for j in range(v1_side):
            src_j = min(j, v1_side - 3)
            for di in range(3):
                for dj in range(3):
                    w[(src_j + dj) * v1_side + src_i + di, j * v1_side + i] = 1
    network.add_connection(
        Connection(
            source=layers[Source],
            target=layers[Target],
            w=w * truncated_normal(size=w.shape),
            wmin=0.,
            wmax=2. * thalamus_to_v1_exc,
            norm=thalamus_to_v1_exc * layers[Source].n,
            update_rule=upd_rl,
            nu=[l_rate * 1e-2, l_rate * 1e-2],
        ),
        source=Source, target=Target
    )

    # V1 EXC -> V1 INH
    Source = 'V1_Exc'
    Target = 'V1_Inh'
    w = torch.zeros(layers[Source].n, layers[Target].n, dtype=torch.float32, device=device)
    for i in range(v1_inh_side):
        src_i = min(i * 2, v1_side - 3)
        for j in range(v1_inh_side):
            src_j = min(j * 2, v1_side - 3)
            for di in range(3):
                for dj in range(3):
                    w[(src_j + dj) * v1_side + src_i + di, j * v1_inh_side + i] = 1
    network.add_connection(
        Connection(
            source=layers[Source],
            target=layers[Target],
            w=w * truncated_normal(size=w.shape),
            wmin=0.,
            wmax=2. * v1_exc_to_v1_inh,
            norm=v1_exc_to_v1_inh * layers[Source].n,
            update_rule=upd_rl,
            nu=[l_rate * 1e-2, l_rate * 1e-2],
        ),
        source=Source, target=Target
    )

    # V1 INH -> V1 EXC
    Source = 'V1_Inh'
    Target = 'V1_Exc'
    w = torch.zeros(layers[Source].n, layers[Target].n, dtype=torch.float32, device=device)
    for i in range(v1_inh_side):
        dst_i = min(i * 2, v1_side - 5)
        for j in range(v1_inh_side):
            dst_j = min(j * 2, v1_side - 5)
            for di in range(5):
                for dj in range(5):
                    w[j * v1_inh_side + i, (dst_j + dj) * v1_side + dst_i + di] = 1
    network.add_connection(
        Connection(
            source=layers[Source],
            target=layers[Target],
            w=w * truncated_normal(size=w.shape),
            wmin=0.,
            wmax=2. * v1_inh_to_v1_exc,
            norm=v1_inh_to_v1_exc * layers[Source].n,
            update_rule=upd_rl,
            nu=[l_rate * 1e-2, l_rate * 1e-2],
        ),
        source=Source, target=Target
    )

    # RECURRENT CONN ON V1
    Source = 'V1_Directions_Input'
    Target = 'V1_Directions_Input'
    w = torch.zeros(layers[Source].n, layers[Target].n, dtype=torch.float32, device=device)
    for i in range(8):
        w[i * v1_neurons:(i + 1) * v1_neurons, i * v1_neurons:(i + 1) * v1_neurons] = 1
    w -= torch.eye(v1_neurons * 8, device=device)
    network.add_connection(
        Connection(
            source=layers[Source],
            target=layers[Target],
            w=w * truncated_normal(size=w.shape),
            wmin=0.,
            wmax=2. * v1_dir_recur,
            norm=v1_dir_recur * layers[Source].n,
            update_rule=upd_rl,
            nu=[l_rate * 1e-2, l_rate * 1e-2],
        ),
        source=Source, target=Target
    )

    all_to_all('V1_Exc', 'Motor', v1_exc_to_motor)
    all_to_all('V1_Directions_Input', 'Motor', v1_dir_to_motor)
    all_to_all('Motor', 'V1_Exc', motor_to_v1_exc)
    all_to_all('Motor', 'V1_Directions_Input', motor_to_v1_dir)
    all_to_all('Motor', 'Motor_Inh', motor_to_motor_inh)
    all_to_all('Motor_Inh', 'Motor', motor_inh_to_motor)

    # Spike recordings for all layers.
    spikes = {}
    for layer in layers:
        spikes[layer] = Monitor(layers[layer], ["s"])  # , time=runtime

    '''
    # Voltage recordings for excitatory and readout layers.
    voltages = {}
    for layer in set(layers.keys()) - {"X"}:
        voltages[layer] = Monitor(layers[layer], ["v"], time=runtime)
    '''

    # Add all monitors to the network.
    for layer in layers:
        network.add_monitor(spikes[layer], name="%s_spikes" % layer)

        # if layer in voltages:
        #    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

    return network, spikes
