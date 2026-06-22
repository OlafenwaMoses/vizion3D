from vizion3d.core.tasks import PROMOTED_TASKS, tasks_by_category


def test_scale_observation_is_promoted_observation_task():
    tasks = tasks_by_category()
    observation = {task.slug: task for task in tasks["Observation"]}
    scale = observation["scale-observation"]
    assert scale.name == "Scale Observation"
    assert scale.status == "promoted"
    assert scale.experimental is False
    assert scale.python_import == "vizion3d.observation.ScaleObservation"
    assert scale.rest_endpoint == "/observation/scale-observation"
    assert scale.grpc_method == "RunScaleObservation"
    assert scale.docs_path == "observation/scale_observation.md"


def test_promoted_tasks_have_unique_slugs():
    slugs = [task.slug for task in PROMOTED_TASKS]
    assert len(slugs) == len(set(slugs))


def test_reconstruction_tasks_are_promoted():
    reconstruction = {task.slug: task for task in tasks_by_category()["Reconstruction"]}
    assert reconstruction["object-3d-reconstruction"].grpc_method == ("RunObject3DReconstruction")
    assert reconstruction["scene-components-3d-reconstruction"].grpc_method == (
        "RunSceneComponents3DReconstruction"
    )
