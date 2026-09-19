from flybrain.drone.simulator import SimplePhysicsSim
from flybrain.drone.physics import DroneCommand, DroneState, clamp_command, THRUST_LIMIT


def test_drone_hovers_with_zero_net_thrust():
    sim = SimplePhysicsSim(DroneState(z=5.0))
    for _ in range(200):
        sim.step(DroneCommand(thrust=0.0), dt=0.01)  # 0 net = cancels gravity
    assert 4.0 < sim.state.z < 6.0


def test_drone_gravity_drops_it():
    sim = SimplePhysicsSim(DroneState(z=5.0))
    for _ in range(300):
        sim.step(DroneCommand(thrust=-9.81), dt=0.01)  # remove thrust entirely
    assert sim.state.z < 5.0


def test_command_clamp():
    c = clamp_command(DroneCommand(thrust=1e6, roll=-1e6, pitch=0.0, yaw=0.0))
    assert c.thrust == THRUST_LIMIT
    assert c.roll == -5.0
