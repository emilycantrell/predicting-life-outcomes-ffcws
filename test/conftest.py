# content of conftest.py
import os


def pytest_addoption(parser):
    parser.addoption(
        "--metadata_concepts",
        action="append",
        help="run all metadata concept related tests ",
    )
    parser.addoption(
        "--metadata_warning", action="append", help="run all metadata tests"
    )
    parser.addoption(
        "--regression", action="append", help="run all linear regression tests"
    )
    parser.addoption(
        "--metalearning", action="append", help="run all metalearning related tests"
    )
    parser.addoption(
        "--imperfect_regression",
        action="append",
        help="run all linear regression tests for which the expected r2_score is less than 1",
    )
    parser.addoption(
        "--incorrect", action="store", help="test all incorrect job desc files"
    )
    parser.addoption(
        "--reproducibility",
        action="store",
        help="two back-to-back runs gives same folds and results",
    )
    parser.addoption(
        "--not_enough_data",
        action="store",
        help="test all job desc files without enough data",
    )
    parser.addoption(
        "--stata",
        action="store",
        help="test loading stata data",
    )


def get_toml_files(dirpath):
    fpaths = []

    def inspect_dir(path):
        for fname in os.listdir(path):
            if fname.endswith("toml"):
                jobdesc = os.path.join(path, fname)
                fpaths.append(jobdesc)

    if isinstance(dirpath, list):
        for path in dirpath:
            inspect_dir(path)
    else:
        inspect_dir(dirpath)
    return fpaths


def pytest_generate_tests(metafunc):
    if "metadata_concepts_toml" in metafunc.fixturenames:
        metafunc.parametrize(
            "metadata_concepts_toml",
            get_toml_files(metafunc.config.getvalue("metadata_concepts")),
        )
    if "metadata_warning_toml" in metafunc.fixturenames:
        metafunc.parametrize(
            "metadata_warning_toml",
            get_toml_files(metafunc.config.getvalue("metadata_warning")),
        )
    if "regression_toml" in metafunc.fixturenames:
        metafunc.parametrize(
            "regression_toml", get_toml_files(metafunc.config.getvalue("regression"))
        )
    if "metalearning_toml" in metafunc.fixturenames:
        metafunc.parametrize(
            "metalearning_toml",
            get_toml_files(metafunc.config.getvalue("metalearning")),
        )
    if "imperfect_regression_toml" in metafunc.fixturenames:
        metafunc.parametrize(
            "imperfect_regression_toml",
            get_toml_files(metafunc.config.getvalue("imperfect_regression")),
        )
    if "incorrect_toml" in metafunc.fixturenames:
        metafunc.parametrize(
            "incorrect_toml", get_toml_files(metafunc.config.getvalue("incorrect"))
        )
    if "not_enough_data_toml" in metafunc.fixturenames:
        metafunc.parametrize(
            "not_enough_data_toml",
            get_toml_files(metafunc.config.getvalue("not_enough_data")),
        )
    if "reproducibility_toml" in metafunc.fixturenames:
        metafunc.parametrize(
            "reproducibility_toml",
            get_toml_files(metafunc.config.getvalue("reproducibility")),
        )
    if "stata_toml" in metafunc.fixturenames:
        metafunc.parametrize(
            "stata_toml",
            get_toml_files(metafunc.config.getvalue("stata")),
        )
