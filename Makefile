.PHONY: test sync-scripts

test:
	python3 -m pytest -q tests

# atomate2-static/scripts/ is the source of the postprocess scripts every package carries.
sync-scripts:
	for package in atomate2-*/; do \
	  [ "$$package" = atomate2-static/ ] || { mkdir -p "$$package/scripts" && cp -p atomate2-static/scripts/report atomate2-static/scripts/plots atomate2-static/scripts/atomate2_postprocess.py "$$package/scripts/"; }; \
	done
