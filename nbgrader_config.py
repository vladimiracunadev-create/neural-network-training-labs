c = get_config()  # noqa: F821
c.CourseDirectory.root = "assignments"
c.CourseDirectory.source_directory = "source"
c.CourseDirectory.release_directory = "release"
c.CourseDirectory.submitted_directory = "submitted"
c.ExecutePreprocessor.timeout = 600
