#include "itkImage.hxx"
#include "itkImageFileReader.hxx"
#include "itkImageSliceIteratorWithIndex.hxx"

#include "caffe/caffe.hpp"

#include "lmdb.h"

#include <iostream>
#include <stdint.h>
#include <sys/stat.h>

DEFINE_string(data, "data",
              "BRATS data folder containing Images and Truth sub-directories.");
DEFINE_string(output, "lmdb-data",
              "Output folder that will contain the LMDB database.");
DEFINE_uint64(kernel, 32,
              "Kernel size.");
DEFINE_uint64(spacing, 32,
              "Distance between kernels.");


int main(int argc, char *argv[])
{
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    typedef itk::Image<unsigned short, 3> mha_image;
    typedef itk::ImageFileReader<mha_image> mha_reader;

    unsigned int kernel_size = FLAGS_kernel;
    unsigned int index_space = FLAGS_spacing;
    unsigned int modalities = 4;

    std::string db_path = FLAGS_output;

    // lmdb
    MDB_env *mdb_env;
    MDB_dbi mdb_dbi;
    MDB_val mdb_key, mdb_data;
    MDB_txn *mdb_txn;

    // Open db
    LOG(INFO) << "Opening lmdb " << db_path;
    CHECK_EQ(mkdir(db_path.c_str(), 0744), 0)
            << "mkdir " << db_path << "failed";
    CHECK_EQ(mdb_env_create(&mdb_env), MDB_SUCCESS) << "mdb_env_create failed";
    CHECK_EQ(mdb_env_set_mapsize(mdb_env, 1099511627776), MDB_SUCCESS)  // 1TB
            << "mdb_env_set_mapsize failed";
    CHECK_EQ(mdb_env_open(mdb_env, db_path.c_str(), 0, 0664), MDB_SUCCESS)
            << "mdb_env_open failed";
    CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
            << "mdb_txn_begin failed";
    CHECK_EQ(mdb_open(mdb_txn, NULL, 0, &mdb_dbi), MDB_SUCCESS)
            << "mdb_open failed. Does the lmdb already exist? ";

    // key for database
    int item_id(0);
    unsigned int num_items = 25;
    int count = 0;

    unsigned int patch_size = kernel_size * kernel_size;
    unsigned int item_size = patch_size * modalities;


    for ( int i = 0 ; i < num_items ; ++ i )
    {
        // SECTION
        // READING ONE IMAGE, 4 MODALITIES, GENERATING KERNELS

        char label;
        char* pixels = new char[item_size];

        std::string img_number = caffe::format_int(i+1, 4);

        std::string input_image_path = FLAGS_data + "/Images/SimBRATS_HG" + img_number + "/";
        std::string truth_image_path = FLAGS_data + "/Truth/";
        std::string t1    = input_image_path + "SimBRATS_HG" + img_number + "_T1.mha";
        std::string t1c   = input_image_path + "SimBRATS_HG" + img_number + "_T1C.mha";
        std::string flair = input_image_path + "SimBRATS_HG" + img_number + "_FLAIR.mha";
        std::string t2    = input_image_path + "SimBRATS_HG" + img_number + "_T2.mha";
        std::string truth = truth_image_path + "SimBRATS_HG" + img_number + "_complete_truth.mha";

        LOG(INFO) << "Processing truth image " << truth;

        // reading input and truth images

        mha_image::Pointer t1_image = mha_image::New();
        mha_image::Pointer t1c_image = mha_image::New();
        mha_image::Pointer t2_image = mha_image::New();
        mha_image::Pointer flair_image = mha_image::New();
        mha_image::Pointer truth_image = mha_image::New();
        mha_reader::Pointer input_image_reader = mha_reader::New();

        try
        {
            input_image_reader->SetFileName(t1);
            input_image_reader->Update();
            t1_image = input_image_reader->GetOutput();
            input_image_reader->SetFileName(t1c);
            input_image_reader->Update();
            t1c_image = input_image_reader->GetOutput();
            input_image_reader->SetFileName(t2);
            input_image_reader->Update();
            t2_image = input_image_reader->GetOutput();
            input_image_reader->SetFileName(flair);
            input_image_reader->Update();
            flair_image = input_image_reader->GetOutput();
            input_image_reader->SetFileName(truth);
            input_image_reader->Update();
            truth_image = input_image_reader->GetOutput();
        }
        catch (itk::ExceptionObject & err)
        {
            LOG(ERROR) << "ExceptionObject caught !";
            LOG(ERROR) << err;
            continue;
        }

        mha_image::RegionType r_t1, r_t1c, r_t2, r_flair, r_truth;
        mha_image::SizeType size;
        mha_image::IndexType index;
        mha_image::PixelType pixel;

        r_t1 = t1_image->GetRequestedRegion();
        r_t1c = t1c_image->GetRequestedRegion();
        r_t2 = t2_image->GetRequestedRegion();
        r_flair = flair_image->GetRequestedRegion();
        r_truth = truth_image->GetRequestedRegion();
        size = r_truth.GetSize();

        int x_size = size[0];
        int y_size = size[1];

        int patches_per_image = 0;
        int empty_patches = 0;

        for ( int start_x_index = 0 ; start_x_index < x_size ; start_x_index += index_space )
        {
            for ( int start_y_index = 0 ; start_y_index < y_size ; start_y_index += index_space )
            {
                LOG(INFO) << "start_x_index: " << start_x_index << " start_y_index: " << start_y_index;

                r_t1.SetSize(0, kernel_size);
                r_t1.SetSize(1, kernel_size);
                r_t1.SetIndex(0, start_x_index);
                r_t1.SetIndex(1, start_y_index);
                //r_t1 = t1_image->GetRequestedRegion();
                r_t1c.SetIndex(0, start_x_index);
                r_t1c.SetIndex(1, start_y_index);
                r_t1c.SetSize(0, kernel_size);
                r_t1c.SetSize(1, kernel_size);
                //r_t1c = t1c_image->GetRequestedRegion();
                r_t2.SetIndex(0, start_x_index);
                r_t2.SetIndex(1, start_y_index);
                r_t2.SetSize(0, kernel_size);
                r_t2.SetSize(1, kernel_size);
                //r_t2 = t2_image->GetRequestedRegion();
                r_flair.SetIndex(0, start_x_index);
                r_flair.SetIndex(1, start_y_index);
                r_flair.SetSize(0, kernel_size);
                r_flair.SetSize(1, kernel_size);
                //r_flair = flair_image->GetRequestedRegion();
                r_truth.SetIndex(0, start_x_index);
                r_truth.SetIndex(1, start_y_index);
                r_truth.SetSize(0, kernel_size);
                r_truth.SetSize(1, kernel_size);
                //r_truth = truth_image->GetRequestedRegion();

                size = r_truth.GetSize();

                itk::ImageSliceIteratorWithIndex<mha_image> t1_iter(t1_image, r_t1);
                t1_iter.SetFirstDirection(0);
                t1_iter.SetSecondDirection(1);
                itk::ImageSliceIteratorWithIndex<mha_image> t1c_iter(t1c_image, r_t1c);
                t1c_iter.SetFirstDirection(0);
                t1c_iter.SetSecondDirection(1);
                itk::ImageSliceIteratorWithIndex<mha_image> t2_iter(t2_image, r_t2);
                t2_iter.SetFirstDirection(0);
                t2_iter.SetSecondDirection(1);
                itk::ImageSliceIteratorWithIndex<mha_image> flair_iter(flair_image, r_flair);
                flair_iter.SetFirstDirection(0);
                flair_iter.SetSecondDirection(1);
                itk::ImageSliceIteratorWithIndex<mha_image> truth_iter(truth_image, r_truth);
                truth_iter.SetFirstDirection(0);
                truth_iter.SetSecondDirection(1);

                std::string value;

                caffe::Datum datum;

                datum.set_channels(modalities);
                datum.set_height(kernel_size);
                datum.set_width(kernel_size);

                t1_iter.GoToBegin();
                t1c_iter.GoToBegin();
                t2_iter.GoToBegin();
                flair_iter.GoToBegin();
                truth_iter.GoToBegin();

                unsigned int x, y, z;

                int patches_per_slice = 0;
                int empty_patches_in_slice = 0;

                while(!truth_iter.IsAtEnd())
                {
                    bool empty = true;

                    while (!truth_iter.IsAtEndOfSlice())
                    {
                        while(!truth_iter.IsAtEndOfLine())
                        {
                            index = truth_iter.GetIndex();
                            x = index[0] - start_x_index;
                            y = index[1] - start_y_index;
                            //z = index[2];
                            pixels[0 * patch_size + y * kernel_size + x] = t1_iter.Get();
                            pixels[1 * patch_size + y * kernel_size + x] = t1c_iter.Get();
                            pixels[2 * patch_size + y * kernel_size + x] = t2_iter.Get();
                            pixels[3 * patch_size + y * kernel_size + x] = flair_iter.Get();
                            label = truth_iter.Get();
                            empty &= (label == 0);

                            ++ t1_iter;
                            ++ t1c_iter;
                            ++ t2_iter;
                            ++ flair_iter;
                            ++ truth_iter;
                        } // end of line

                        t1_iter.NextLine();
                        t1c_iter.NextLine();
                        t2_iter.NextLine();
                        flair_iter.NextLine();
                        truth_iter.NextLine();
                    } // end of slice

                    // pass to the next slice

                    t1_iter.NextSlice();
                    t1c_iter.NextSlice();
                    t2_iter.NextSlice();
                    flair_iter.NextSlice();
                    truth_iter.NextSlice();

                    // and save the patch in lmdb if not empty

                    if ( empty )
                    {
                        ++ empty_patches_in_slice;
                        continue;
                    }

                    datum.set_data(pixels, item_size);
                    datum.set_label(label);
                    std::string key_str = caffe::format_int(item_id, 8);
                    datum.SerializeToString(&value);
                    ++ item_id;

                    // Put in db
                    mdb_data.mv_size = value.size();
                    mdb_data.mv_data = reinterpret_cast<void*>(&value[0]);
                    mdb_key.mv_size = key_str.size();
                    mdb_key.mv_data = reinterpret_cast<void*>(&key_str[0]);
                    CHECK_EQ(mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0), MDB_SUCCESS)
                            << "mdb_put failed";

                    if (++count % 1000 == 0) {
                        // Commit txn
                        CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS)
                                << "mdb_txn_commit failed";
                        CHECK_EQ(mdb_txn_begin(mdb_env, NULL, 0, &mdb_txn), MDB_SUCCESS)
                                << "mdb_txn_begin failed";
                    }

                    ++ patches_per_slice;

                } // end

                patches_per_image += patches_per_slice;
                empty_patches += empty_patches_in_slice;

            } // x-y patch

            LOG(INFO) << "Processed " << patches_per_image << " image patch.";
            LOG(INFO) << "Processed " << empty_patches << " empty image patch.";

        } // x patches

        delete[] pixels;

    } // num_items

    LOG(INFO) << "Processed " << num_items << " images.";

    // END OF SECTION

    // write the last batch
    if (count % 1000 != 0) {
        CHECK_EQ(mdb_txn_commit(mdb_txn), MDB_SUCCESS) << "mdb_txn_commit failed";
        mdb_close(mdb_env, mdb_dbi);
        mdb_env_close(mdb_env);
        LOG(INFO) << "Processed " << count << " patches.";
    }

    return 0;
}
