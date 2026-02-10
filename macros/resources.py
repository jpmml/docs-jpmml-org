import textwrap

def define_env(env):

	@env.macro
	def resources_section():
		page = env.variables.get("page", {})

		repo = "jpmml/docs-jpmml-org"

		resources = []

		notebook = page.meta.get("notebook", None)
		if notebook:
			view_link = "https://nbviewer.jupyter.org/github/{}/blob/master/notebooks/{}".format(repo, notebook)
			download_link = "https://github.com/{}/raw/master/notebooks/{}".format(repo, notebook)

			resources.append(("Notebook", "[View]({}) or [Download]({})".format(view_link, download_link)))

		project = page.meta.get("project", None)
		if project:
			view_link = "https://github.com/{}/tree/master/projects/{}".format(repo, project)
			download_link = "https://download-directory.github.io/?" + view_link

			resources.append(("Project template", "[View]({}) or [Download]({})".format(view_link, download_link)))

		if resources:
			resources_list = "\n".join("* {}: {}".format(name, links) for name, links in resources)

			md_string = """
			## Resources
			
			{}
			""".format(resources_list)
			
			return textwrap.dedent(md_string)
		else:
			return ""
