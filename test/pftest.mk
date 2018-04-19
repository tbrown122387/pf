##
## Auto Generated makefile by CodeLite IDE
## any manual changes will be erased      
##
## Debug
ProjectName            :=pftest
ConfigurationName      :=Debug
WorkspacePath          :=/home/taylor/Documents/ssmworkspace
ProjectPath            :=/home/taylor/pf/test
IntermediateDirectory  :=Debug
OutDir                 := $(IntermediateDirectory)
CurrentFileName        :=
CurrentFilePath        :=
CurrentFileFullPath    :=
User                   :=taylor
Date                   :=17/04/18
CodeLitePath           :=/home/taylor/.codelite
LinkerName             :=/usr/bin/x86_64-linux-gnu-g++
SharedObjectLinkerName :=/usr/bin/x86_64-linux-gnu-g++ -shared -fPIC
ObjectSuffix           :=.o
DependSuffix           :=.o.d
PreprocessSuffix       :=.i
DebugSwitch            :=-g 
IncludeSwitch          :=-I
LibrarySwitch          :=-l
OutputSwitch           :=-o 
LibraryPathSwitch      :=-L
PreprocessorSwitch     :=-D
SourceSwitch           :=-c 
OutputFile             :=$(IntermediateDirectory)/$(ProjectName)
Preprocessors          :=
ObjectSwitch           :=-o 
ArchiveOutputSwitch    := 
PreprocessOnlySwitch   :=-E
ObjectsFileList        :="pftest.txt"
PCHCompileFlags        :=
MakeDirCommand         :=mkdir -p
LinkOptions            :=  
IncludePath            :=  $(IncludeSwitch). $(IncludeSwitch). $(IncludeSwitch)/usr/local/include/UnitTest++ $(IncludeSwitch)/usr/include/eigen3 $(IncludeSwitch)../include 
IncludePCH             := 
RcIncludePath          := 
Libs                   := $(LibrarySwitch)UnitTest++ $(LibrarySwitch)pf 
ArLibs                 :=  "libUnitTest++.a" "libpf.a" 
LibPath                := $(LibraryPathSwitch). $(LibraryPathSwitch)/usr/local/lib $(LibraryPathSwitch)../Release/ 

##
## Common variables
## AR, CXX, CC, AS, CXXFLAGS and CFLAGS can be overriden using an environment variables
##
AR       := /usr/bin/x86_64-linux-gnu-ar rcu
CXX      := /usr/bin/x86_64-linux-gnu-g++
CC       := /usr/bin/x86_64-linux-gnu-gcc
CXXFLAGS :=  -g -std=c++11 $(Preprocessors)
CFLAGS   :=  -g $(Preprocessors)
ASFLAGS  := 
AS       := /usr/bin/x86_64-linux-gnu-as


##
## User defined environment variables
##
CodeLiteDir:=/usr/share/codelite
Objects0=$(IntermediateDirectory)/main.cpp$(ObjectSuffix) $(IntermediateDirectory)/test_rv_eval.cpp$(ObjectSuffix) $(IntermediateDirectory)/test_resamplers.cpp$(ObjectSuffix) $(IntermediateDirectory)/test_utils.cpp$(ObjectSuffix) $(IntermediateDirectory)/test_rv_samp.cpp$(ObjectSuffix) 



Objects=$(Objects0) 

##
## Main Build Targets 
##
.PHONY: all clean PreBuild PrePreBuild PostBuild MakeIntermediateDirs
all: $(OutputFile)

$(OutputFile): $(IntermediateDirectory)/.d $(Objects) 
	@$(MakeDirCommand) $(@D)
	@echo "" > $(IntermediateDirectory)/.d
	@echo $(Objects0)  > $(ObjectsFileList)
	$(LinkerName) $(OutputSwitch)$(OutputFile) $(Objects) $(LibPath) $(Libs) $(LinkOptions)

MakeIntermediateDirs:
	@test -d Debug || $(MakeDirCommand) Debug


$(IntermediateDirectory)/.d:
	@test -d Debug || $(MakeDirCommand) Debug

PreBuild:


##
## Objects
##
$(IntermediateDirectory)/main.cpp$(ObjectSuffix): main.cpp $(IntermediateDirectory)/main.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/taylor/pf/test/main.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/main.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/main.cpp$(DependSuffix): main.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/main.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/main.cpp$(DependSuffix) -MM main.cpp

$(IntermediateDirectory)/main.cpp$(PreprocessSuffix): main.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/main.cpp$(PreprocessSuffix) main.cpp

$(IntermediateDirectory)/test_rv_eval.cpp$(ObjectSuffix): test_rv_eval.cpp $(IntermediateDirectory)/test_rv_eval.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/taylor/pf/test/test_rv_eval.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/test_rv_eval.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/test_rv_eval.cpp$(DependSuffix): test_rv_eval.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/test_rv_eval.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/test_rv_eval.cpp$(DependSuffix) -MM test_rv_eval.cpp

$(IntermediateDirectory)/test_rv_eval.cpp$(PreprocessSuffix): test_rv_eval.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/test_rv_eval.cpp$(PreprocessSuffix) test_rv_eval.cpp

$(IntermediateDirectory)/test_resamplers.cpp$(ObjectSuffix): test_resamplers.cpp $(IntermediateDirectory)/test_resamplers.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/taylor/pf/test/test_resamplers.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/test_resamplers.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/test_resamplers.cpp$(DependSuffix): test_resamplers.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/test_resamplers.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/test_resamplers.cpp$(DependSuffix) -MM test_resamplers.cpp

$(IntermediateDirectory)/test_resamplers.cpp$(PreprocessSuffix): test_resamplers.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/test_resamplers.cpp$(PreprocessSuffix) test_resamplers.cpp

$(IntermediateDirectory)/test_utils.cpp$(ObjectSuffix): test_utils.cpp $(IntermediateDirectory)/test_utils.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/taylor/pf/test/test_utils.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/test_utils.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/test_utils.cpp$(DependSuffix): test_utils.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/test_utils.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/test_utils.cpp$(DependSuffix) -MM test_utils.cpp

$(IntermediateDirectory)/test_utils.cpp$(PreprocessSuffix): test_utils.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/test_utils.cpp$(PreprocessSuffix) test_utils.cpp

$(IntermediateDirectory)/test_rv_samp.cpp$(ObjectSuffix): test_rv_samp.cpp $(IntermediateDirectory)/test_rv_samp.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/taylor/pf/test/test_rv_samp.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/test_rv_samp.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/test_rv_samp.cpp$(DependSuffix): test_rv_samp.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/test_rv_samp.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/test_rv_samp.cpp$(DependSuffix) -MM test_rv_samp.cpp

$(IntermediateDirectory)/test_rv_samp.cpp$(PreprocessSuffix): test_rv_samp.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/test_rv_samp.cpp$(PreprocessSuffix) test_rv_samp.cpp


-include $(IntermediateDirectory)/*$(DependSuffix)
##
## Clean
##
clean:
	$(RM) -r Debug/


