#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <cassert>

#include <arrow/api.h>
#include <plasma/client.h>
#include <plasma/common.h>
#include <arrow/util/logging.h>
#include <arrow/util/key_value_metadata.h>
#include <arrow/tensor.h>
#include <arrow/array.h>
#include <arrow/table.h>
#include <arrow/buffer.h>
#include <arrow/type_fwd.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/writer.h>
#include <arrow/ipc/reader.h>
#include <arrow/datum.h>
#include <arrow/compute/api_scalar.h>
#include <arrow/compute/api_vector.h>
#include <arrow/chunked_array.h>

#include <boost/date_time/date.hpp>
#include <boost/date_time.hpp>

typedef std::vector<std::shared_ptr<arrow::Table>> TableVector;

const int min_height{200}, max_height{800}, hours{12};
const uint8_t time_resolution{1}, height_resolution{5};

const int height_dim = (max_height-min_height) / height_resolution;
const int time_dim = hours * 60 / time_resolution;

int64_t to_unixtime(boost::posix_time::ptime t) {
    auto diff = t - boost::posix_time::ptime(boost::gregorian::date(1970, boost::gregorian::Jan, 1));
    return diff.total_seconds();
}

TableVector read_tables(plasma::PlasmaClient& client) {
    plasma::ObjectTable objTable;
    ARROW_CHECK_OK(client.List(&objTable));
    std::vector<plasma::ObjectID> obj_ids(objTable.size());
    std::size_t i{0};
    for (auto const& [obj_id, entry] : objTable)
        obj_ids[i++] = obj_id;
    std::vector<plasma::ObjectBuffer> buffers;
    ARROW_CHECK_OK(client.Get(obj_ids, -1, &buffers));
    std::vector<arrow::io::BufferReader> buffer_readers;
    TableVector tables;
    for(const auto& buffer : buffers) {
        auto buffer_reader = std::make_shared<arrow::io::BufferReader>(buffer.data);
        auto result = arrow::ipc::RecordBatchStreamReader::Open(buffer_reader);
        if(result.ok()) {
            auto stream_reader = result.ValueOrDie();
            auto result = arrow::Table::FromRecordBatchReader(&*stream_reader);
            if(result.ok()) {
                auto table = result.ValueOrDie();
                tables.emplace_back(table);
            }
            else {
                ARROW_LOG(ERROR) << result.status();
            }
        }
        else {
            ARROW_LOG(ERROR) << result.status();
        }
    }
    ARROW_CHECK_OK(client.Delete(obj_ids));
    return tables;
}

std::vector<int8_t> flatten(std::vector<std::vector<int8_t>>& v) {
    size_t total_size = 0;
    for(auto& sub : v)
        total_size += sub.size();
    std::vector<int8_t> result;
    result.reserve(total_size);
    for (auto& sub : v)
        result.insert(result.end(), sub.begin(), sub.end());
    return result;
}

std::pair<size_t, size_t> compute_indices(int64_t time, int64_t height, boost::posix_time::ptime start, boost::posix_time::ptime end) {
    auto start_unix = to_unixtime(start);
    auto end_unix = to_unixtime(end);
    // std::cout << "start_unix " << start_unix << " current time " << time << std::endl;
    // std::cout << "difference " << time - start_unix << std::endl;
    size_t time_idx = (time - start_unix) / (time_resolution * 60);
    size_t height_idx = (height - min_height) / height_resolution;
    return {time_idx, height_idx};
}

std::pair<std::vector<std::string>, std::vector<std::shared_ptr<arrow::ChunkedArray>>> read_arrays(const std::shared_ptr<arrow::Table>& table) {
    auto datetime_col = table->GetColumnByName(std::string("datetime"));
    auto heights_col = table->GetColumnByName(std::string("GDALT"));
    const auto l = datetime_col->length();
    int64_t year;
    {
        auto _date = datetime_col->GetScalar(l/2).ValueOrDie();
        auto _year = *static_cast<int64_t *>(
                        const_cast<arrow::Int64Scalar&>(
                            arrow::compute::Year(arrow::Datum(_date)).ValueOrDie().scalar_as<arrow::Int64Scalar>())
                                    .mutable_data());
        year = _year;
    }
    std::vector<std::shared_ptr<arrow::ChunkedArray>> chunked_arrays;
    std::vector<std::string> date_strs;
    {
        namespace bg = boost::gregorian;
        namespace bpt = boost::posix_time;
        namespace ac = arrow::compute;
        bg::day_iterator day_itr(bg::date(year, bg::Jan, 1));
        auto _end_date = bg::date(year, bg::Dec, 31);
        size_t i{0};
        do {
            // std::cout << to_simple_string(*day_itr) << std::endl;
            auto day = day_itr->day();
            auto month = day_itr->month();
            
            // auto second_unit = ;
            // second_unit()
            auto SECONDS = arrow::timestamp(arrow::TimeUnit::SECOND);
            arrow::TimestampScalar x(0, SECONDS);
            arrow::TimestampScalar y(1, SECONDS);
            // std::cout << x.ToString() << " " << y.ToString() << std::endl;
            auto _some_date = bg::date(2020, 10, 30);
            arrow::TimestampScalar z(to_unixtime(bpt::ptime(_some_date)), SECONDS);
            // std::cout << "special date " << z.ToString() << std::endl;
            // std::cout << "x value: " << x.value << std::endl;
            auto start_t = bpt::ptime(bg::date(year, month, day), bpt::time_duration(19, 0, 0));
            auto end_t = bpt::ptime(bg::date(year, month, day) + bg::days(1), bpt::time_duration(7, 0, 0));
            
            auto start_t_unix = to_unixtime(start_t);
            auto end_t_unix = to_unixtime(end_t);

            std::vector<bool> bitmap(l, false);
            int count{0};
            for(; i < l; ++i) {
                auto height = std::static_pointer_cast<arrow::DoubleScalar>(heights_col->GetScalar(i).ValueOrDie())->value;
                if(height < min_height || height > max_height) continue;

                int64_t t_unix = std::static_pointer_cast<arrow::TimestampScalar>(datetime_col->GetScalar(i).ValueOrDie())->value / 1e9;
                if(t_unix < start_t_unix) continue;
                if(t_unix >= end_t_unix) break;
                if(t_unix < end_t_unix) bitmap[i] = true, ++count;
            }
            if(count) {
                std::cout << "number of observations between " << bpt::to_simple_string(start_t) << " and " << bpt::to_simple_string(end_t) << " is: " << count << std::endl;
                arrow::BooleanBuilder builder;
                ARROW_CHECK_OK(builder.Reserve(l));
                ARROW_CHECK_OK(builder.AppendValues(bitmap));
                auto array = builder.Finish().ValueOrDie();
                auto filtered = ac::Filter(arrow::Datum(table), arrow::Datum(array)).ValueOrDie().table();
                auto filtered_datetime = filtered->GetColumnByName(std::string("datetime"));
                auto filtered_heights = filtered->GetColumnByName(std::string("GDALT"));
                auto filtered_snr = filtered->GetColumnByName(std::string("SNL"));
                std::vector<std::vector<int8_t>> mesh(time_dim, std::vector<int8_t>(height_dim));
                std::vector<std::vector<bool>> valid(time_dim, std::vector<bool>(height_dim, false));
                for(size_t j = 0; j < filtered->num_rows(); ++j) {
                    int64_t time = std::static_pointer_cast<arrow::TimestampScalar>(filtered_datetime->GetScalar(j).ValueOrDie())->value / 1e9;
                    // std::cout << "START TIME: " << bpt::to_simple_string(start_t) << std::endl;
                    // std::cout << "CURRENT TIME: " << std::static_pointer_cast<arrow::TimestampScalar>(filtered_datetime->GetScalar(j).ValueOrDie())->ToString() << std::endl;
                    // std::cout << "END TIME: " << bpt::to_simple_string(end_t) << std::endl;
                    int64_t height = std::static_pointer_cast<arrow::DoubleScalar>(filtered_heights->GetScalar(j).ValueOrDie())->value;
                    int8_t snr = round(std::static_pointer_cast<arrow::DoubleScalar>(filtered_snr->GetScalar(j).ValueOrDie())->value);
                    // std::cout << "computing indices of " << height << " and " << time << std::endl;
                    const auto& [t, h] = compute_indices(time, height, start_t, end_t);
                    // std::cout << "t " << t << " h " << std::endl;
                    mesh[t][h] = snr;
                    valid[t][h] = true;
                }
                arrow::ArrayVector chunks;
                for(int i = 0; i < mesh.size(); ++i) {
                    auto& mesh_row = mesh[i];
                    auto& validity_row = valid[i];
                    arrow::Int8Builder builder;
                    ARROW_CHECK_OK(builder.Reserve(mesh_row.size()));
                    ARROW_CHECK_OK(builder.AppendValues(mesh_row, validity_row));
                    auto chunk = builder.Finish().ValueOrDie();
                    chunks.emplace_back(chunk);
                }
                auto chunked_array = arrow::ChunkedArray::Make(chunks, arrow::int8()).ValueOrDie();
                chunked_arrays.push_back(chunked_array);
                auto current_date = bpt::ptime(bg::date(year, month, day));
                date_strs.emplace_back(bpt::to_simple_string(current_date));
                // auto x = filtered_datetime->GetScalar(0).ValueOrDie();
                // std::cout << x->ToString() << std::endl;
            }
        } while(++day_itr <= _end_date);
    }
    return {date_strs, chunked_arrays};
}



int main() {
    plasma::PlasmaClient client;
    ARROW_CHECK_OK(client.Connect("/tmp/plasma", "", 0));
    auto tables = read_tables(client);
    for(const auto& table: tables)
        for(const auto& field : table->fields())
            std::cout << field->ToString() << std::endl;
    for(const auto& table : tables) {
        const auto& [date_strs, chunked_arrays] = read_arrays(table);
        // TODO: schema
        std::vector<std::shared_ptr<arrow::Field>> fields(date_strs.size());
        for(int i = 0; i < fields.size(); ++i) {
            auto& field = fields[i];
            auto& date_str = date_strs[i];
            field = arrow::field(date_str, arrow::int8());
        }
        auto metadata = arrow::KeyValueMetadata::Make({"min_height", "max_height", "hours", "time_resolution", "height_resolution"}, 
                                                        {std::to_string(min_height), std::to_string(max_height), std::to_string(hours), 
                                                            std::to_string(time_resolution), std::to_string(height_resolution)});
        auto schema = arrow::schema(fields, metadata);
        auto new_table = arrow::Table::Make(schema, chunked_arrays);
    }
    ARROW_CHECK_OK(client.Disconnect());
}


















// #include <arrow/api.h>
// #include <plasma/client.h>
// #include <arrow/util/logging.h>
// #include <arrow/tensor.h>
// #include <arrow/array.h>
// #include <arrow/buffer.h>
// #include <arrow/io/memory.h>
// #include <arrow/ipc/writer.h>
// using namespace plasma;
// using namespace arrow;

// int main(int argc, char** argv) {
//   // Start up and connect a Plasma client.
//   PlasmaClient client;
//   ARROW_CHECK_OK(client.Connect("/tmp/plasma", "", 0));
//   // Generate an Object ID for Plasma
//   ObjectID object_id = ObjectID::from_binary("11111111111111111111");

//   // Generate Float Array
//   int64_t input_length = 1000;
//   std::vector<float> input(input_length);
//   for (int64_t i = 0; i < input_length; ++i) {
//     input[i] = 2.0;
//   }

//   // Create Arrow Tensor Object, no copy made!
//   // {input_length} is the shape of the tensor
//   auto value_buffer = Buffer::Wrap<float>(input);
//   Tensor t(float32(), value_buffer, {input_length});

//   // Get the size of the tensor to be stored in Plasma
//   int64_t datasize;
//   ARROW_CHECK_OK(ipc::GetTensorSize(t, &datasize));
//   int32_t meta_len = 0;

//   // Create the Plasma Object
//   // Plasma is responsible for initializing and resizing the buffer
//   // This buffer will contain the _serialized_ tensor
//   std::shared_ptr<Buffer> buffer;
//   ARROW_CHECK_OK(
//       client.Create(object_id, datasize, NULL, 0, &buffer));

//   // Writing Process, this will copy the tensor into Plasma
//   io::FixedSizeBufferWriter stream(buffer);
//   ARROW_CHECK_OK(arrow::ipc::WriteTensor(t, &stream, &meta_len, &datasize));

//   // Seal Plasma Object
//   // This computes a hash of the object data by default
//   ARROW_CHECK_OK(client.Seal(object_id));
// }