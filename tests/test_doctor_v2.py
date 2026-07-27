from neural_labs.doctor import environment_doctor


def test_doctor_returns_core_readiness() -> None:
    report = environment_doctor()
    assert "checks" in report
    assert "optional_packages" in report["checks"]
    assert isinstance(report["ready_for_core_labs"], bool)
