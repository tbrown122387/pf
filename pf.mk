##
## Auto Generated makefile by CodeLite IDE
## any manual changes will be erased      
##
## Release
ProjectName            :=pf
ConfigurationName      :=Release
WorkspacePath          :=/home/taylor/Documents/ssmworkspace
ProjectPath            :=/home/taylor/pf
IntermediateDirectory  :=./Release
OutDir                 := $(IntermediateDirectory)
CurrentFileName        :=
CurrentFilePath        :=
CurrentFileFullPath    :=
User                   :=taylor
Date                   :=19/04/18
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
OutputFile             :=$(IntermediateDirectory)/lib$(ProjectName).a
Preprocessors          :=
ObjectSwitch           :=-o 
ArchiveOutputSwitch    := 
PreprocessOnlySwitch   :=-E
ObjectsFileList        :="pf.txt"
PCHCompileFlags        :=
MakeDirCommand         :=mkdir -p
LinkOptions            :=  
IncludePath            :=  $(IncludeSwitch). $(IncludeSwitch). $(IncludeSwitch)/usr/include/eigen3 $(IncludeSwitch)./include 
IncludePCH             := 
RcIncludePath          := 
Libs                   := 
ArLibs                 :=  
LibPath                := $(LibraryPathSwitch). 

##
## Common variables
## AR, CXX, CC, AS, CXXFLAGS and CFLAGS can be overriden using an environment variables
##
AR       := /usr/bin/x86_64-linux-gnu-ar rcu
CXX      := /usr/bin/x86_64-linux-gnu-g++
CC       := /usr/bin/x86_64-linux-gnu-gcc
CXXFLAGS :=  -pg -O3 -std=c++11 $(Preprocessors)
CFLAGS   :=   $(Preprocessors)
ASFLAGS  := 
AS       := /usr/bin/x86_64-linux-gnu-as


##
## User defined environment variables
##
CodeLiteDir:=/usr/share/codelite
Objects0=$(IntermediateDirectory)/src_rv_samp.cpp$(ObjectSuffix) $(IntermediateDirectory)/src_rv_eval.cpp$(ObjectSuffix) 



Objects=$(Objects0) 

##
## Main Build Targets 
##
.PHONY: all clean PreBuild PrePreBuild PostBuild MakeIntermediateDirs
all: $(IntermediateDirectory) $(OutputFile)

$(OutputFile): $(Objects)
	@$(MakeDirCommand) $(@D)
	@echo "" > $(IntermediateDirectory)/.d
	@echo $(Objects0)  > $(ObjectsFileList)
	$(AR) $(ArchiveOutputSwitch)$(OutputFile) $(Objects) $(ArLibs)
	@$(MakeDirCommand) "/home/taylor/Documents/ssmworkspace/.build-release"
	@echo rebuilt > "/home/taylor/Documents/ssmworkspace/.build-release/pf"

MakeIntermediateDirs:
	@test -d ./Release || $(MakeDirCommand) ./Release


./Release:
	@test -d ./Release || $(MakeDirCommand) ./Release

PreBuild:


##
## Objects
##
$(IntermediateDirectory)/src_rv_samp.cpp$(ObjectSuffix): src/rv_samp.cpp $(IntermediateDirectory)/src_rv_samp.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/taylor/pf/src/rv_samp.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_rv_samp.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_rv_samp.cpp$(DependSuffix): src/rv_samp.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_rv_samp.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/src_rv_samp.cpp$(DependSuffix) -MM src/rv_samp.cpp

$(IntermediateDirectory)/src_rv_samp.cpp$(PreprocessSuffix): src/rv_samp.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_rv_samp.cpp$(PreprocessSuffix) src/rv_samp.cpp

$(IntermediateDirectory)/src_rv_eval.cpp$(ObjectSuffix): src/rv_eval.cpp $(IntermediateDirectory)/src_rv_eval.cpp$(DependSuffix)
	$(CXX) $(IncludePCH) $(SourceSwitch) "/home/taylor/pf/src/rv_eval.cpp" $(CXXFLAGS) $(ObjectSwitch)$(IntermediateDirectory)/src_rv_eval.cpp$(ObjectSuffix) $(IncludePath)
$(IntermediateDirectory)/src_rv_eval.cpp$(DependSuffix): src/rv_eval.cpp
	@$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) -MG -MP -MT$(IntermediateDirectory)/src_rv_eval.cpp$(ObjectSuffix) -MF$(IntermediateDirectory)/src_rv_eval.cpp$(DependSuffix) -MM src/rv_eval.cpp

$(IntermediateDirectory)/src_rv_eval.cpp$(PreprocessSuffix): src/rv_eval.cpp
	$(CXX) $(CXXFLAGS) $(IncludePCH) $(IncludePath) $(PreprocessOnlySwitch) $(OutputSwitch) $(IntermediateDirectory)/src_rv_eval.cpp$(PreprocessSuffix) src/rv_eval.cpp


-include $(IntermediateDirectory)/*$(DependSuffix)
##
## Clean
##
clean:
	$(RM) -r ./Release/


