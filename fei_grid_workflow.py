from multiprocessing import Pool
import os
import shutil
import errno
import glob
import shlex
import pickle
import json
import datetime
import subprocess
import tarfile
import math

import b2luigi as luigi
from b2luigi.basf2_helper.tasks import Basf2PathTask
from b2luigi.basf2_helper.utils import get_basf2_git_hash
from b2luigi.batch.processes.gbasf2 import run_with_gbasf2

from B_generic_train import create_fei_path, get_particles
import fei


def force_symlink(source, target):
    try:
        os.symlink(source, target)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(target)
            os.symlink(source, target)


# ballpark CPU times for the FEI stages executed on grid. Format: <stage-number> : <minutes>
grid_cpu_time = {
    -1: 10,
    0: 20,
    1: 30,
    2: 40,
    3: 4 * 60,  # adapted to event-based processing. Usual time per file: 8 hours
    4: 4 * 60,  # adapted to event-based processing. Usual time per file: 16 hours
    5: 4 * 60,  # adapted to event-based processing. Usual time per file: more than 36 hours
    6: 4 * 60,  # adapted to event-based processing. Usual time per file: more than 48 hours
}

processing_type = {
    -1: {"type": "file_based"},
    0: {"type": "file_based"},
    1: {"type": "file_based"},
    2: {"type": "file_based"},
    3: {"type": "event_based", "n_events": 100000},  # usually 1/2 of a file
    4: {"type": "event_based", "n_events": 50000},  # usually 1/4 of a file
    5: {"type": "event_based", "n_events": 20000},  # usually 1/10 of a file
    6: {"type": "event_based", "n_events": 10000},  # usually 1/20 of a file
}

fei_analysis_outputs = {}
fei_analysis_outputs[-1] = ["mcParticlesCount.root"]
for i in range(6):
    fei_analysis_outputs[i] = ["training_input.root"]
fei_analysis_outputs[6] = [
    "Monitor_FSPLoader.root",
    "Monitor_Final.root",
    "Monitor_ModuleStatistics.root",
    "Monitor_PostReconstruction_AfterMVA.root",
    "Monitor_PostReconstruction_AfterRanking.root",
    "Monitor_PostReconstruction_BeforePostCut.root",
    "Monitor_PostReconstruction_BeforeRanking.root",
    "Monitor_PreReconstruction_AfterRanking.root",
    "Monitor_PreReconstruction_AfterVertex.root",
    "Monitor_PreReconstruction_BeforeRanking.root",
]


class FEIAnalysisSummaryTask(luigi.Task):

    gbasf2_project_name_prefix = luigi.Parameter(significant=False)
    gbasf2_input_dslist = luigi.Parameter(hashed=True, significant=False)

    cache = luigi.IntParameter(significant=False)
    monitor = luigi.BoolParameter(significant=False)
    stage = luigi.IntParameter()
    mode = luigi.Parameter()

    # should be a json file with directories to outputs of FEIAnalysisTask instances
    def output(self):

        yield self.add_to_output("list_of_output_directories.json")

    # creates a separate FEIAnalysisTask for each dataset (line) in the dslist
    # this is done under assumption,
    # that an input dataset does not have more than 1000 files == number of jobs
    def requires(self):

        dslistfile = open(self.gbasf2_input_dslist, 'r')
        dslist = [dsname.strip() for dsname in dslistfile.readlines()]
        dslistfile.close()

        # Creating datbase of input files as json file (lfn, nEvents)
        files_database_name = os.path.join(os.path.dirname(self.gbasf2_input_dslist), 'files_database.json')
        print("Obtaining information on input files...")
        files_database = {}
        if not os.path.isfile(files_database_name):

            for ds in dslist:
                files_pattern = ""
                dsname = ""
                dspathlist = ds.split('/')
                if dspathlist[-1].endswith('.root'):  # Catch case, where file with structure /*/.../*/sub*/*.root is given
                    dsname = '/'.join(dspathlist[:-2])
                    files_pattern = ds
                elif dspathlist[-1].startswith('sub'):  # Catch case, where dataset with structure /*/.../*/sub* is given
                    dsname = '/'.join(dspathlist[:-1])
                    files_pattern = '/'.join([dsname, 'sub*/*.root'])
                else:  # assume here, that the case with the dataset name is given without sub*
                    dsname = ds
                    files_pattern = '/'.join([dsname, 'sub*/*.root'])

                proc = run_with_gbasf2(shlex.split(f"gb2_ds_query_file {files_pattern} -m nEvents,lfn"), capture_output=True)
                for info in proc.stdout.strip().splitlines():
                    infolist = info.strip().split('|')[1:]

                    # Store info as dict for file with values for {lfn: nEvents}
                    nEvents = int(infolist[-2].split(':')[-1].strip())
                    lfn = infolist[-1].split(':')[-1].strip()
                    files_database.setdefault(dsname, {})[lfn] = nEvents

            with open(files_database_name, 'w') as filesdb:
                json.dump(files_database, filesdb, sort_keys=True, indent=2)

        else:

            with open(files_database_name, 'r') as filesdb:
                files_database = json.load(filesdb)

        print(f"Information on input files stored in {files_database_name}.")

        flists_foldername = 'partial_filelists'
        flist_dir = os.path.dirname(self.gbasf2_input_dslist)
        flist_path = os.path.join(flist_dir, flists_foldername)
        if not os.path.isdir(flist_path):
            os.makedirs(flist_path)

        if processing_type[self.stage]["type"] == "file_based":

            for index, dataset in enumerate(dslist):

                partial_dslistname, extension = os.path.splitext(self.gbasf2_input_dslist)
                partial_dslistname += f"_{self.stage}_Part{index}" + extension
                partial_dslistpath = os.path.join(flist_path, os.path.basename(partial_dslistname))

                # Make sure, that a proper partial file list is created for that particular stage
                if not os.path.isfile(partial_dslistpath):
                    partial_dslist = open(partial_dslistpath, 'w')
                    partial_dslist.write(dataset)
                    partial_dslist.close()

                yield FEIAnalysisTask(
                    cache=self.cache,
                    monitor=self.monitor,
                    mode=f"{self.mode}_Part{index}",
                    stage=self.stage,
                    gbasf2_project_name_prefix=luigi.get_setting("gbasf2_project_name_prefix") + f"_Part{index}",
                    gbasf2_input_dslist=partial_dslistpath,
                )

        elif processing_type[self.stage]["type"] == "event_based":

            index = 0
            for ds in dslist:

                max_events = 0
                dsname = ""
                dspathlist = ds.split('/')
                if dspathlist[-1].endswith('.root'):  # Catch case, where file with structure /*/.../*/sub*/*.root is given
                    dsname = '/'.join(dspathlist[:-2])
                    max_events = max(set([nEvents for lfn, nEvents in files_database[dsname].items() if lfn == ds]))
                elif dspathlist[-1].startswith('sub'):  # Catch case, where dataset with structure /*/.../*/sub* is given
                    dsname = '/'.join(dspathlist[:-1])
                    max_events = max(set(files_database[dsname].values()))
                else:  # assume here, that the case with the dataset name is given without sub*
                    dsname = ds
                    max_events = max(set(files_database[dsname].values()))

                parts_per_ds = math.ceil(max_events / float(processing_type[self.stage]["n_events"]))
                for dspart in range(parts_per_ds):

                    partial_dslistname, extension = os.path.splitext(self.gbasf2_input_dslist)
                    partial_dslistname += f"_{self.stage}_Part{index}" + extension
                    partial_dslistpath = os.path.join(flist_path, os.path.basename(partial_dslistname))

                    # Make sure, that a proper partial file list is created for that particular stage
                    if not os.path.isfile(partial_dslistpath):
                        partial_dslist = open(partial_dslistpath, 'w')
                        partial_dslist.write(ds)
                        partial_dslist.close()

                    yield FEIAnalysisTask(
                        cache=self.cache,
                        monitor=self.monitor,
                        mode=f"{self.mode}_Part{index}",
                        stage=self.stage,
                        gbasf2_project_name_prefix=luigi.get_setting("gbasf2_project_name_prefix") + f"_Part{index}",
                        gbasf2_input_dslist=partial_dslistpath,
                        process_events=processing_type[self.stage]["n_events"],
                        skip_events=dspart*processing_type[self.stage]["n_events"],
                    )

                    index += 1

    def run(self):

        outputs = {}
        for inname in fei_analysis_outputs[self.stage]:
            outputs[inname] = []
            for folder in self.get_input_file_names(inname):
                outputs[inname] += glob.glob(os.path.join(folder, "*.root"))

        with open(self.get_output_file_name("list_of_output_directories.json"), 'w') as jsonfile:
            json.dump(outputs, jsonfile, sort_keys=True, indent=2)


class FEIAnalysisTask(Basf2PathTask):

    batch_system = "gbasf2"

    git_hash = luigi.Parameter(hashed=True, default=get_basf2_git_hash(), significant=False)

    gbasf2_project_name_prefix = luigi.Parameter(significant=False)
    gbasf2_basf2opt = luigi.Parameter(significant=False, default=luigi.get_setting("gbasf2_basf2opt"))
    gbasf2_input_dslist = luigi.Parameter(hashed=True, significant=False)

    cache = luigi.IntParameter(significant=False)
    monitor = luigi.BoolParameter(significant=False)
    stage = luigi.IntParameter()
    mode = luigi.Parameter()

    skip_events = luigi.IntParameter(significant=False, default=0)
    process_events = luigi.IntParameter(significant=False, default=0)

    def output(self):

        for outname in fei_analysis_outputs[self.stage]:
            yield self.add_to_output(outname)

    def requires(self):

        if self.stage == -1:

            return []  # default implementation, as in the luigi.Task (if no requirements are present)

        else:

            # need the timestamp of the additional inputs to build their TMP-SE path
            yield PrepareInputsTask(
                mode="AnalysisInput",
                stage=self.stage - 1,
                remote_tmp_directory=luigi.get_setting("remote_tmp_directory"),
                remote_initial_se=luigi.get_setting("remote_initial_se"),
            )

            # need a symlink to the merged mcParticlesCount.root file
            yield MergeOutputsTask(
                mode="Merging",
                stage=-1,
                ncpus=luigi.get_setting("local_cpus"),
            )

            # need symlinks to *.xml files of FEI training of previous stages
            for fei_stage in range(self.stage):

                yield FEITrainingTask(
                    mode="Training",
                    stage=fei_stage,
                )

    def create_path(self):

        luigi.set_setting("gbasf2_cputime", grid_cpu_time[self.stage])

        if processing_type[self.stage]["type"] == "event_based":
            self.gbasf2_basf2opt += f" -n {self.process_events} --skip-events {self.skip_events}"

        # determine the remote TMP-SE destination of input tarball for gbasf2 command
        if self.stage > -1:
            timestamp = open(f"{self.get_input_file_names('successful_input_upload.txt')[0]}", "r").read().strip()
            additional_file = os.path.join(luigi.get_setting("remote_tmp_directory").rstrip('/')+timestamp,
                                           "stage"+str(self.stage - 1), "sub00", "fei_analysis_inputs.tar.gz")
            luigi.set_setting("gbasf2_input_datafiles", [additional_file])

            # create symlinks to files, which are needed for current FEI analysis stage
            for key in self.get_input_file_names():
                if key == "mcParticlesCount.root" or key.endswith(".xml"):
                    force_symlink(self.get_input_file_names(key)[0], key)

        path = create_fei_path(filelist=[], cache=self.cache, monitor=self.monitor)

        if self.stage > -1:
            # remove symlinks and not needed Summary.pickle files
            for key in self.get_input_file_names():
                if key == "mcParticlesCount.root" or key.endswith(".xml"):
                    os.remove(key)
            for summary_file in glob.glob("Summary.pickle*"):
                os.remove(summary_file)
        return path


class MergeOutputsTask(luigi.Task):

    ncpus = luigi.IntParameter(significant=False)  # to be used with setting 'local_cpus' in settings.json
    stage = luigi.IntParameter()
    mode = luigi.Parameter()

    def output(self):

        for outname in fei_analysis_outputs[self.stage]:
            yield self.add_to_output(outname)

    def requires(self):

        cache = -1 if self.stage == -1 else 0
        monitor = True if self.stage == 6 else False
        yield FEIAnalysisSummaryTask(
            cache=cache,
            monitor=monitor,
            mode="TrainingInput",
            stage=self.stage,
            gbasf2_project_name_prefix=luigi.get_setting("gbasf2_project_name_prefix"),
            gbasf2_input_dslist=luigi.get_setting("gbasf2_input_dslist"),
        )

    def run(self):

        cmds = []
        outputs = {}
        with open(self.get_input_file_names("list_of_output_directories.json")[0], 'r') as jsonfile:
            outputs = json.load(jsonfile)

        for inname in fei_analysis_outputs[self.stage]:
            cmds.append(f"analysis-fei-mergefiles {self.get_output_file_name(inname)} " +
                        " ".join(outputs[inname]))

        p = Pool(self.ncpus)
        p.map(subprocess.check_call, [shlex.split(cmd) for cmd in cmds])


class FEITrainingTask(luigi.Task):

    stage = luigi.IntParameter()
    mode = luigi.Parameter()
    first_xml_output = None

    def output(self):

        if self.stage == -1:
            yield self.add_to_output("dataset_sites.txt")
        elif self.stage < 6:
            # load particles to determine .xml output names
            particles = get_particles()
            myparticles = fei.core.get_stages_from_particles(particles)
            for p in myparticles[self.stage]:
                for channel in p.channels:
                    if not self.first_xml_output:
                        self.first_xml_output = f"{channel.label}.xml"
                    yield self.add_to_output(f"{channel.label}.xml")
        elif self.stage == 6:
            # outputs from report scripts
            yield self.add_to_output("summary.tex")
            yield self.add_to_output("summary.txt")

            # outputs resulting from mva evaluation
            yield self.add_to_output("training_input_merged.root")
            particles = get_particles()
            myparticles = fei.core.get_stages_from_particles(particles)
            for stage in range(self.stage):
                for p in myparticles[stage]:
                    for channel in p.channels:
                        yield self.add_to_output(f"{channel.label}.zip")

    def requires(self):

        # need merged mcParticlesCount.root from stage -1
        yield MergeOutputsTask(
            mode="Merging",
            stage=-1,
            ncpus=luigi.get_setting("local_cpus"),
        )

        # need merged training_input.root or merged Monitor files from current stage
        if self.stage > -1:

            yield MergeOutputsTask(
                mode="Merging",
                stage=self.stage,
                ncpus=luigi.get_setting("local_cpus"),
            )

        # need .xml training files from all previous stages of FEI training,
        # beginning with stage 1
        if self.stage > 0:

            for fei_stage in range(self.stage):

                yield FEITrainingTask(
                    mode="Training",
                    stage=fei_stage,
                )

        # need merged training_input.root from stages 0 to 5 for mva evaluation
        if self.stage == 6:

            for fei_stage in range(self.stage):

                yield MergeOutputsTask(
                    mode="Merging",
                    stage=fei_stage,
                    ncpus=luigi.get_setting("local_cpus"),
                )

    def run(self):

        if self.stage == -1:

            input_ds = luigi.get_setting("gbasf2_input_dslist")
            input_dslist = []
            if input_ds.endswith('.txt'):
                input_dslist = [line.strip() for line in open(input_ds, 'r').readlines()]
            else:
                input_dslist = [input_ds]

            # Extracting site names from gb2_ds_list query
            print("Obtaining sites of input files...")
            proc_stdouts = []
            for ds in input_dslist:
                proc = run_with_gbasf2(shlex.split(f"gb2_ds_list {ds} -lg"), capture_output=True)
                proc_stdouts.append(proc.stdout.splitlines())

            sites = []
            for stdout in proc_stdouts:
                sites += [siteline.split(':')[0].replace('DATA', 'TMP') for siteline in stdout if 'SE' in siteline]
            sites = list(set(sites))
            with open(f"{self.get_output_file_name('dataset_sites.txt')}", 'w') as output_sites:
                output_sites.write('\n'.join(sites))
                output_sites.close()

        else:

            # create symlinks to files, which are needed for current FEI analysis stage
            for key in self.get_input_file_names():
                if key == "mcParticlesCount.root" or key == "training_input.root" or "Monitor" in key or key.endswith(".xml"):
                    force_symlink(self.get_input_file_names(key)[0], key)

            monitor = True if self.stage == 6 else False
            if self.stage < 6:
                # load path to perform training
                for summary_file in glob.glob("Summary.pickle*"):
                    os.remove(summary_file)
                if not os.path.exists('Summary.pickle'):
                    create_fei_path(filelist=[], cache=0, monitor=monitor)
                particles, configuration = pickle.load(open('Summary.pickle', 'rb'))
                fei.do_trainings(particles, configuration)
            else:
                cmds = []
                for summary_file in glob.glob("Summary.pickle*"):
                    os.remove(summary_file)
                if not os.path.exists('Summary.pickle'):
                    create_fei_path(filelist=[], cache=0, monitor=monitor)

                # create report outputs
                printReporting = os.path.join(os.getenv("BELLE2_LOCAL_DIR"), "analysis/scripts/fei/printReporting.py")
                latexReporting = os.path.join(os.getenv("BELLE2_LOCAL_DIR"), "analysis/scripts/fei/latexReporting.py")
                cmds.append(f"basf2 {printReporting} > {self.get_output_file_name('summary.txt')}")
                cmds.append(f"basf2 {latexReporting} {self.get_output_file_name('summary.tex')}")
                retcodes = [subprocess.call(cmd, shell=True) for cmd in cmds]

                for png in glob.glob("*.png"):
                    shutil.move(png, os.path.join(os.path.dirname(self.get_output_file_name('summary.tex')), png))

                # prepare and preform mva evaluation
                mva_cmds = []
                valid_trainings = []
                invalid_trainings = []
                particles = get_particles()
                myparticles = fei.core.get_stages_from_particles(particles)
                for stage in range(self.stage):
                    for p in myparticles[stage]:
                        for channel in p.channels:
                            if not fei.core.Teacher.check_if_weightfile_is_fake(f"{channel.label}.xml"):
                                valid_trainings.append(channel.label)
                            else:
                                invalid_trainings.append(channel.label)

                if valid_trainings:
                    individual_training_inputs = " ".join(self.get_input_file_names("training_input.root"))
                    mva_cmds.append(f"analysis-fei-mergefiles {self.get_output_file_name('training_input_merged.root')} " +
                                    individual_training_inputs)
                    for label in valid_trainings:
                        mva_cmds.append(f"basf2_mva_evaluate.py -i '{label}.xml' "
                                        f"--data {self.get_output_file_name('training_input_merged.root')} "
                                        f"--treename '{label} variables' "
                                        f"-o '{self.get_output_file_name(label+'.zip')}'")
                    retcodes += [subprocess.call(cmd, shell=True) for cmd in mva_cmds]

                else:
                    f = open(self.get_output_file_name('training_input_merged.root'), 'w')
                    f.truncate(0)
                    f.close()

                # if non-zero error code, output files probably corrupt, so removing them
                if sum(retcodes) != 0:
                    if os.path.exists(self.get_output_file_name('summary.txt')):
                        os.remove(self.get_output_file_name('summary.txt'))
                    if os.path.exists(self.get_output_file_name('summary.tex')):
                        os.remove(self.get_output_file_name('summary.tex'))
                    for png in glob.glob(os.path.join(os.path.dirname(self.get_output_file_name('summary.tex')), "*.png")):
                        os.remove(png)

                    if os.path.exists(self.get_output_file_name('training_input_merged.root')):
                        os.remove(self.get_output_file_name('training_input_merged.root'))

                    for label in valid_trainings:
                        if os.path.exists(self.get_output_file_name(label+'.zip')):
                            os.remove(self.get_output_file_name(label+'.zip'))

                else:
                    for label in invalid_trainings:
                        f = open(self.get_output_file_name(label+'.zip'), 'w')
                        f.truncate(0)
                        f.close()

            # remove symlinks and not needed Summary.pickle files
            for key in self.get_input_file_names():
                if key == "mcParticlesCount.root" or key == "training_input.root" or "Monitor" in key or key.endswith(".xml"):
                    os.remove(key)
            for summary_file in glob.glob("Summary.pickle*"):
                os.remove(summary_file)

            if self.stage < 6:

                # determine directory of outputs:
                outputdir = os.path.dirname(self.get_output_file_name(self.first_xml_output))

                # move *.xml and *.log files to output directory
                for fpath in glob.glob("*.xml"):
                    shutil.move(fpath, outputdir)
                for fpath in glob.glob("*.log"):
                    shutil.move(fpath, outputdir)


class PrepareInputsTask(luigi.Task):

    remote_tmp_directory = luigi.Parameter(significant=False)  # should be set via settings
    remote_initial_se = luigi.Parameter(significant=False)  # should be set via settings

    stage = luigi.IntParameter()
    mode = luigi.Parameter()

    def output(self):

        yield self.add_to_output('fei_analysis_inputs.tar.gz')
        yield self.add_to_output('successful_input_upload.txt')

    def requires(self):

        # need merged mcParticlesCount.root from stage -1
        yield MergeOutputsTask(
            mode="Merging",
            stage=-1,
            ncpus=luigi.get_setting("local_cpus"),
        )

        # need dataset_sites.txt file from stage -1 of FEI training,
        # which is (ab)used to create that site list
        yield FEITrainingTask(
            mode="Training",
            stage=-1,
        )

        # need .xml training files from current stage of FEI training
        if self.stage > -1:
            yield FEITrainingTask(
                mode="Training",
                stage=self.stage,
            )

        # need .xml training files from all previous stages of FEI training,
        # beginning with stage 1
        if self.stage > 0:

            for fei_stage in range(self.stage):

                yield FEITrainingTask(
                    mode="Training",
                    stage=fei_stage,
                )

    def run(self):

        # create tarball with all required input files for FEIAnalysisTask
        outputs = [outs[0] for outs in self.get_input_file_names().values()
                   if outs[0].endswith('.root') or outs[0].endswith('.xml')]
        taroutname = self.get_output_file_name("fei_analysis_inputs.tar.gz")
        taroutdir = os.path.dirname(taroutname)

        if os.path.isfile(self.get_output_file_name('successful_input_upload.txt')):
            os.remove(self.get_output_file_name('successful_input_upload.txt'))

        tar = tarfile.open(taroutname, "w:gz")
        for output in outputs:
            tar.add(output, arcname=os.path.basename(output))
        tar.close()

        # upload tarball to initial storage element
        timestamp = datetime.datetime.now().strftime("_%b-%d-%Y_%H-%M-%S")
        foldername = os.path.join(self.remote_tmp_directory.rstrip('/')+timestamp, "stage"+str(self.stage))
        completed_copy = run_with_gbasf2(shlex.split(f"gb2_ds_put -d {self.remote_initial_se} "
                                         f"-i {taroutdir} --datablock sub00 {foldername}"))

        # replicate tarball to other storage element sites (defined by used input datasets)
        dataset_sites = [site.strip() for site in open(self.get_input_file_names('dataset_sites.txt')[0], 'r').readlines()]
        completed_replicas = []
        for ds_site in dataset_sites:
            completed_replicas.append(run_with_gbasf2(shlex.split(f"gb2_ds_rep {foldername}/"
                                      f"sub00 -d {ds_site} -s {self.remote_initial_se} --force")))

        if sum([proc.returncode for proc in completed_replicas + [completed_copy]]) == 0:
            with open(f"{self.get_output_file_name('successful_input_upload.txt')}", "w") as timestampfile:
                timestampfile.write(timestamp)


class ProduceStatisticsTask(luigi.WrapperTask):

    def requires(self):

        yield FEITrainingTask(
            mode="Training",
            stage=6,
        )

        # yield MergeOutputsTask(
        #     mode="Merging",
        #     stage=6,
        #     ncpus=luigi.get_setting("local_cpus"),
        # )

        # yield PrepareInputsTask(
        #     mode="AnalysisInput",
        #     stage=0,
        #     remote_tmp_directory=luigi.get_setting("remote_tmp_directory"),
        #     remote_initial_se=luigi.get_setting("remote_initial_se"),
        # )

        # yield FEIAnalysisSummaryTask(
        #     cache=0,
        #     monitor=True,
        #     mode="TrainingInput",
        #     stage=6,
        #     gbasf2_project_name_prefix=luigi.get_setting("gbasf2_project_name_prefix"),
        #     gbasf2_input_dslist=luigi.get_setting("gbasf2_input_dslist"),
        # )


if __name__ == '__main__':
    main_task_instance = ProduceStatisticsTask()
    dslist = luigi.get_setting("gbasf2_input_dslist")
    n_gbasf2_tasks = len(open(dslist, 'r').readlines()) * 20
    luigi.process(main_task_instance, workers=n_gbasf2_tasks)
